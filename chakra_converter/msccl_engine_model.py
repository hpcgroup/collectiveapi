"""MSCCL runtime engine facts that the MSCCL -> Chakra converter needs.

Source: torchcomms-msccl v2_28, comms/ncclx/v2_28/src (called "rt" below). Every count here is a
source formula. Nothing is fitted.

1. Stripe calls (converter flag --msccl_stripe_calls)

   The engine splits each IR step of a rank into nStripes passes when the rank's send pressure
   before its first receive is larger than the FIFO depth (rt/msccl_engine_host.cc:166-212):

       pressure(TB) = sum(count * chunkSteps) over the TB's send steps before its first recv step
       nStripes     = ceil(max pressure / NCCL_STEPS) if max pressure > NCCL_STEPS, else 1

   The kernel walks the whole program once per stripe (rt/device/msccl_engine.cuh:285-333). In
   stripe k a step of `count` chunks moves floor(count*(k+1)/S) - floor(count*k/S) chunks. A
   consumer waits for its producer only in the first stripe, and the producer publishes its flag
   only after the last stripe (cuh:295-317, 415-423). So the calls of one step sit in series on
   the critical path, and each call pays the full per-message cost.

   The converter emits one send per IR step. With the flag on, a send that stands for m > 1
   stripe calls gets (m - 1) more per-message costs as send delay. The per-message cost is what
   the emitted message already pays: the route latency plus the send's own protocol delay.
   No new number enters the model. A send with m = 1 is unchanged.

2. Connection tags (converter flag --msccl_conn_tags)

   The converter tags every message with its TB channel. AstraSim matches a receive to a send
   by issue order per (tag, src, dst, size), so two TBs of one rank that share
   (channel, peer, direction) share one FIFO, and a receive can complete on the other TB's data.
   The runtime gives each such TB its own connection slot connIndex, counted in TB order per
   (channel, peer, direction), only for TBs that use that direction
   (rt/msccl_engine_host.cc:246-312). With the flag on, tag = channel + 64 * connIndex.
   connIndex 0 keeps tag = channel, so IRs without shared connections convert unchanged.

3. runtime_v3 launch stripe count, R4a (converter option msccl_stripe_runtime="v3")

   rt3 = torchcomms-msccl branch krydzy/runtime-v3 at 3228413, comms/ncclx/v2_28/src (the build runtime_v3,
   defaults NCCL_MSCCL_STRIPE_COUNT=launch, NCCL_MSCCL_STRIPE_FLAGS=auto).

   Parts 1 and 2 describe runtime_v2b: nStripes is fixed per rank when the algorithm is added, from
   count * chunkSteps (n_stripes below). runtime_v3 computes it per launch (rt3/msccl_engine_host.cc:560-583):
   the kernel issues one prims call per maxAllowedCount chunks, and a call takes chunkSteps FIFO slots
   whatever its size (rt3/device/msccl_engine.cuh:359-362), so

       maxAllowedCount = clamp(chunkBytes / (msgBytes / nChunksPerLoop), 1, 71)        rt3 host.cc:863-871
       pressure(TB)    = sum(DIVUP(count, maxAllowedCount) * chunkSteps) over the TB's send steps up to
                         and including its first recv-type step                        rt3 host.cc:150-164
       nStripes        = max over ALL ranks of min(64, ceil(pressure / NCCL_STEPS) if pressure > NCCL_STEPS
                         else 1)                                                       rt3 host.cc:166-171, 619-628

   Every rank uses the same value for one message size (rt3 host.cc:873-881). The 64 is the default of
   NCCL_MSCCL_MAX_STRIPES (rt3 host.cc:90); it clips, it does not reject.
   The runtime has an MSCCL launch only for AllReduce (ncclAllReduceMscclImpl, rt3 host.cc:830), so the
   launch formula is defined for AllReduce IRs only.

4. runtime_v3 per-stripe dependency flags, R4b (same option; used by the Stage 5b stripe passes)

   runtime_v2b (part 1): a dependency is waited for in stripe 0 only, on a flag that the producer publishes
   after its last stripe. Where the dependencies of a rank then form a cycle, the launch never ends.
   runtime_v3 can publish a flag at every stripe (rt3/device/msccl_engine.cuh:434) and marks dependencies as
   "aligned" (mscclStripeAnalyze, rt3 host.cc:245-481): an aligned dependency waits at every stripe k for the
   producer's stripe k; any other waits at stripe 0 for the producer's last stripe, as before
   (rt3 cuh:305-332).
   With the default NCCL_MSCCL_STRIPE_FLAGS=auto a rank uses the per-stripe flags only when its launch has
   nStripes > 1, the runtime_v2b flag order has a cycle and the marked order has none (rt3 host.cc:886-895;
   the analysis runs per rank, on that rank's TBs, rt3 host.cc:748-762). Every other rank and launch keeps
   runtime_v2b's order. stripe_flags() below returns that choice and the marks.
"""

NCCL_STEPS = 8                  # rt/include/device.h:24
MSCCL_ENGINE_MAX_TBS = 64       # rt/include/msccl_engine.h:30; channel < 64 (host.cc:254)
SIMPLE_CHUNKSTEPS = 4           # rt/include/msccl_engine.h:45

# Step types that use the send / receive side of a connection (rt/msccl_engine_host.cc:263-275).
SEND_TYPES = frozenset({"s", "rcs", "rrs", "rrcs"})
RECV_TYPES = frozenset({"r", "rcs", "rrs", "rrc", "rrcs"})


def chunk_steps(protocol):
    """FIFO slots per chunk (rt/msccl_engine_host.cc:64-66)."""
    return SIMPLE_CHUNKSTEPS if protocol == "Simple" else 1


def n_stripes(gpu_xml, protocol):
    """nStripes of one rank (rt/msccl_engine_host.cc:166-212)."""
    steps_per_chunk = chunk_steps(protocol)
    max_pressure = 0
    for tb in gpu_xml.findall("tb"):
        if int(tb.attrib.get("chan", -1)) < 0:
            continue
        pressure = 0
        for step in sorted(tb.findall("step"), key=lambda s: int(s.attrib["s"])):
            step_type = step.attrib["type"]
            if step_type in SEND_TYPES:
                pressure += int(step.attrib["cnt"]) * steps_per_chunk
            if step_type in RECV_TYPES:
                break
        max_pressure = max(max_pressure, pressure)
    if max_pressure <= NCCL_STEPS:
        return 1
    return -(-max_pressure // NCCL_STEPS)


def stripe_calls(count, stripes):
    """Stripes in which a step of `count` chunks moves at least one chunk (cuh:330-333)."""
    return sum(1 for k in range(stripes) if (count * (k + 1)) // stripes - (count * k) // stripes > 0)


def conn_index(gpu_xml):
    """Runtime connIndex per TB: {tb id: (recv slot, send slot)} (rt/msccl_engine_host.cc:246-312)."""
    next_recv, next_send, slots = {}, {}, {}
    for tb in sorted(gpu_xml.findall("tb"), key=lambda e: int(e.get("id", 0))):
        tb_id = int(tb.get("id"))
        chan = int(tb.get("chan", -1))
        if chan < 0:
            slots[tb_id] = (0, 0)
            continue
        types = [s.get("type", "nop") for s in tb.findall("step")]
        recv_peer, send_peer = int(tb.get("recv", -1)), int(tb.get("send", -1))
        recv_slot = send_slot = 0
        if recv_peer >= 0 and any(t in RECV_TYPES for t in types):
            recv_slot = next_recv.get((chan, recv_peer), 0)
            next_recv[(chan, recv_peer)] = recv_slot + 1
        if send_peer >= 0 and any(t in SEND_TYPES for t in types):
            send_slot = next_send.get((chan, send_peer), 0)
            next_send[(chan, send_peer)] = send_slot + 1
        slots[tb_id] = (recv_slot, send_slot)
    return slots


def conn_tag(chan, slot):
    """AstraSim tag of connection slot `slot` on channel `chan`. Injective because chan < 64."""
    if not 0 <= chan < MSCCL_ENGINE_MAX_TBS:
        raise ValueError(f"channel {chan} is outside [0, {MSCCL_ENGINE_MAX_TBS}): tags would collide")
    return chan + MSCCL_ENGINE_MAX_TBS * slot


# ---------------------------------------------------------------------------------------------------------
# runtime_v3 (rt3) launch values. One Python implementation, shared with the evaluator's executability gate.
MSCCL_ENGINE_MAX_COUNT = 72     # rt3/include/msccl_engine.h:31; maxAllowedCount <= 71 (rt3 host.cc:869)
MSCCL_ENGINE_MAX_STRIPES = 64   # rt3/include/msccl_engine.h:43; default of NCCL_MSCCL_MAX_STRIPES (rt3 host.cc:90)
# chunkBytes = buffSizes[proto] / NCCL_STEPS (LL: / 2 for the data half of a line) * chunkSteps
# (rt3 host.cc:73-88); buffer defaults Simple 4 MiB, LL 512 KiB (NCCL_BUFFSIZE is not set on the HW runs).
CHUNK_BYTES = {"Simple": (4 << 20) // NCCL_STEPS * SIMPLE_CHUNKSTEPS, "LL": (512 << 10) // NCCL_STEPS // 2}


def rank_tbs(gpu_xml):
    """One rank as the engine sees it: TBs in id order, each dict(send, recv, chan, steps), with step tuples
    (type, srcbuf, srcoff, dstbuf, dstoff, cnt, dep TB index, dep step, hasdep) in step order."""
    tb_xml = sorted(gpu_xml.findall("tb"), key=lambda e: int(e.get("id", 0)))
    index = {int(t.get("id", 0)): b for b, t in enumerate(tb_xml)}
    tbs = []
    for t in tb_xml:
        steps = []
        for s in sorted(t.findall("step"), key=lambda e: int(e.get("s", 0))):
            depid = int(s.get("depid", -1))
            steps.append((s.get("type", "nop"), s.get("srcbuf", "i"), int(s.get("srcoff", 0)),
                          s.get("dstbuf", "o"), int(s.get("dstoff", 0)), int(s.get("cnt", 1)),
                          index.get(depid, depid) if depid >= 0 else -1, int(s.get("deps", -1)),
                          int(s.get("hasdep", 0))))
        tbs.append(dict(send=int(t.get("send", -1)), recv=int(t.get("recv", -1)), chan=int(t.get("chan", -1)),
                        steps=steps))
    return tbs


def send_pressure(tbs, steps_per_chunk, merge=1):
    """mscclSendPressure (rt3 host.cc:150-164): max over TBs with a channel of the FIFO slots of the send
    steps up to and including the first recv-type step. merge = 1: the AddAlgo formula (runtime_v2b, part 1);
    merge = maxAllowedCount: the merged calls of one launch."""
    max_pressure = 0
    for tb in tbs:
        if tb["chan"] < 0:
            continue
        pressure = 0
        for step in tb["steps"]:
            if step[0] in SEND_TYPES:
                pressure += -(-step[5] // merge) * steps_per_chunk
            if step[0] in RECV_TYPES:
                break
        max_pressure = max(max_pressure, pressure)
    return max_pressure


def stripes_from_pressure(pressure, cap=MSCCL_ENGINE_MAX_STRIPES):
    """mscclStripesFromPressure (rt3 host.cc:166-171); cap = NCCL_MSCCL_MAX_STRIPES."""
    stripes = -(-pressure // NCCL_STEPS) if pressure > NCCL_STEPS else 1
    if cap >= 1 and stripes > cap:
        stripes = cap
    return stripes


def max_allowed_count(protocol, msg_bytes, nchunksperloop):
    """Chunks per prims call of one launch (rt3 host.cc:863-871)."""
    chunk_msg_bytes = msg_bytes // nchunksperloop if nchunksperloop > 0 else 0
    if chunk_msg_bytes <= 0:
        return 1
    chunk_bytes = CHUNK_BYTES.get(protocol, CHUNK_BYTES["Simple"])
    return max(1, min(MSCCL_ENGINE_MAX_COUNT - 1, chunk_bytes // chunk_msg_bytes))


def launch_stripes(ranks, protocol, msg_bytes, nchunksperloop, cap=MSCCL_ENGINE_MAX_STRIPES):
    """nStripes of one launch with NCCL_MSCCL_STRIPE_COUNT=launch: the max over all ranks, so the same on
    every rank (rt3 host.cc:619-628, 873-881). ranks: rank_tbs() of every rank of the IR."""
    merge = max_allowed_count(protocol, msg_bytes, nchunksperloop)
    steps_per_chunk = chunk_steps(protocol)
    return max([1] + [stripes_from_pressure(send_pressure(tbs, steps_per_chunk, merge), cap) for tbs in ranks])


# ---------------------------------------------------------------------------------------------------------
# libstdc++ std::sort (bits/stl_algo.h: introsort, threshold 16, median-of-3 pivot, heap fallback,
# final insertion sort). Its order of equal keys decides which edge mscclStripeAnalyze demotes first.
def _std_sort(a, key):
    n = len(a)
    if n < 2:
        return

    def lt(x, y):
        return key(x) < key(y)

    def adjust_heap(first, hole, length, value):
        top = hole
        second = hole
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if lt(a[first + second], a[first + second - 1]):
                second -= 1
            a[first + hole] = a[first + second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            a[first + hole] = a[first + second - 1]
            hole = second - 1
        parent = (hole - 1) // 2
        while hole > top and lt(a[first + parent], value):
            a[first + hole] = a[first + parent]
            hole = parent
            parent = (hole - 1) // 2
        a[first + hole] = value

    def heap_sort(first, last):
        length = last - first
        if length >= 2:
            parent = (length - 2) // 2
            while True:
                adjust_heap(first, parent, length, a[first + parent])
                if parent == 0:
                    break
                parent -= 1
        while last - first > 1:
            last -= 1
            value = a[last]
            a[last] = a[first]
            adjust_heap(first, 0, last - first, value)

    def move_median_to_first(result, x, y, z):
        if lt(a[x], a[y]):
            if lt(a[y], a[z]):
                a[result], a[y] = a[y], a[result]
            elif lt(a[x], a[z]):
                a[result], a[z] = a[z], a[result]
            else:
                a[result], a[x] = a[x], a[result]
        elif lt(a[x], a[z]):
            a[result], a[x] = a[x], a[result]
        elif lt(a[y], a[z]):
            a[result], a[z] = a[z], a[result]
        else:
            a[result], a[y] = a[y], a[result]

    def unguarded_partition(first, last, pivot):
        while True:
            while lt(a[first], a[pivot]):
                first += 1
            last -= 1
            while lt(a[pivot], a[last]):
                last -= 1
            if not first < last:
                return first
            a[first], a[last] = a[last], a[first]
            first += 1

    def introsort(first, last, depth):
        while last - first > 16:
            if depth == 0:
                heap_sort(first, last)
                return
            depth -= 1
            mid = first + (last - first) // 2
            move_median_to_first(first, first + 1, mid, last - 1)
            cut = unguarded_partition(first + 1, last, first)
            introsort(cut, last, depth)
            last = cut

    def linear_insert(last):
        val = a[last]
        nxt = last - 1
        while lt(val, a[nxt]):
            a[last] = a[nxt]
            last = nxt
            nxt -= 1
        a[last] = val

    def insertion_sort(first, last):
        for i in range(first + 1, last):
            if lt(a[i], a[first]):
                val = a[i]
                a[first + 1:i + 1] = a[first:i]
                a[first] = val
            else:
                linear_insert(i)

    introsort(0, n, 2 * (n.bit_length() - 1))
    if n > 16:
        insertion_sort(0, 16)
        for i in range(16, n):
            linear_insert(i)
    else:
        insertion_sort(0, n)



# ---------------------------------------------------------------------------------------------------------
# R4b alignment analysis: port of mscclStripeAnalyze (rt3 host.cc:245-481), line for line, including the order
# in which it demotes edges (so the marks equal the runtime's).
def _accesses(st):
    """mscclStepAccesses (rt3 host.cc:177-194): (buf, write, off, cnt); input and output are one buffer."""
    t, c = st[0], st[5]
    if c <= 0:
        return ()
    rd = (1 if st[1] == "s" else 0, 0, st[2], c)
    wr = (1 if st[3] == "s" else 0, 1, st[4], c)
    if t in ("s", "rrs"):
        return (rd,)
    if t in ("r", "rcs"):
        return (wr,)
    if t in ("rrc", "rrcs", "cpy", "re"):
        return (rd, wr)
    return ()


def _stripe_of(q, n, S):
    """mscclStripeOf (rt3 host.cc:198-200): the stripe that moves position q of a step of n chunks."""
    return ((q + 1) * S + n - 1) // n - 1


def stripe_analyze(tbs, S, auto=False):
    """Returns dict(cycle_step, cycle, aligned=set((bid, step)), skipped=reason|None, ...) for one rank.
    pub(j) = j: the codegen emits numDependencies <= 1 and numReductions = 1 (ir_codegen_interp.py:214-250).
    auto=True: stop once cycle_step is known to be 0 (NCCL_MSCCL_STRIPE_FLAGS=auto then uses step flags and
    never reads the marks; same decision as the full analysis, which computes cycle_step before any demotion)."""
    res = dict(cycle_step=0, cycle=0, aligned=set(), skipped=None, demoted=0, intra=0)
    nB = len(tbs)
    nSt = [len(tb["steps"]) for tb in tbs]
    edges = []                               # [c, i, p, j, al]
    inE = [[[] for _ in range(n)] for n in nSt]
    for b, tb in enumerate(tbs):
        for i, st in enumerate(tb["steps"]):
            p, ds = st[6], st[7]
            if p < 0 or ds < 0:
                continue
            if p >= nB or p == b:
                res["skipped"] = "dep block out of range"
                return res
            jstar = -1
            psteps = tbs[p]["steps"]
            for j in range(max(0, ds), nSt[p]):
                if psteps[j][8]:
                    jstar = j
                    break
            if jstar < 0:
                res["skipped"] = "dep never published"
                return res
            inE[b][i].append(len(edges))
            edges.append([b, i, p, jstar, 0])
    evU = sum(nSt)
    if evU * S * nB > 32 * 1024 * 1024:
        res["skipped"] = "vector-clock budget"
        return res
    if auto:
        # cheap pre-pass with the full analysis's early exits, in its order: unstriped dependency cycle,
        # access budget, then cycle_step (the step-order flag graph)
        if not _topo_ok(nSt, inE, edges):
            res["skipped"] = "dependency cycle"
            return res
        nrec = 0
        for tb in tbs:
            for st in tb["steps"]:
                nrec += sum(a[3] for a in _accesses(st))
                if nrec > 16 * 1024 * 1024:
                    res["skipped"] = "access budget"
                    return res
        if not _striped_cycle(nSt, inE, edges, S):
            return res

    # unstriped HB
    vcU = [[-1] * (n * nB) for n in nSt]
    ptr = [0] * nB
    done = 0
    progress = True
    while progress:
        progress = False
        for b in range(nB):
            while ptr[b] < nSt[b]:
                i = ptr[b]
                ready = True
                for e in inE[b][i]:
                    if ptr[edges[e][2]] <= edges[e][3]:
                        ready = False
                        break
                if not ready:
                    break
                v = vcU[b]
                o = i * nB
                if i > 0:
                    v[o:o + nB] = v[o - nB:o]
                for e in inE[b][i]:
                    E = edges[e]
                    u = vcU[E[2]]
                    uo = E[3] * nB
                    for x in range(nB):
                        if u[uo + x] > v[o + x]:
                            v[o + x] = u[uo + x]
                v[o + b] = i
                ptr[b] += 1
                done += 1
                progress = True
    if done != evU:
        res["skipped"] = "dependency cycle"
        return res

    def hbU(xa, ia, xb, ib):
        return vcU[xb][ib * nB + xa] >= ia

    # access records, sorted by key with libstdc++'s order
    recs = []
    for b, tb in enumerate(tbs):
        for i, st in enumerate(tb["steps"]):
            for (buf, w, off, cnt) in _accesses(st):
                for q in range(cnt):
                    recs.append(((buf << 32) | ((off + q) & 0xFFFFFFFF), b, i, q, cnt, w))
            if len(recs) > 16 * 1024 * 1024:
                res["skipped"] = "access budget"
                return res
    _std_sort(recs, key=lambda r: r[0])

    # candidates
    for E in edges:
        c, i, p, j = E[0], E[1], E[2], E[3]
        csteps = tbs[c]["steps"]
        k = i
        while k < nSt[c] and csteps[k][0] == "nop":
            k += 1
        if k >= nSt[c]:
            continue
        pa = _accesses(tbs[p]["steps"][j])
        ca = _accesses(csteps[k])
        conflict, ok = False, True
        for x in pa:
            if not ok:
                break
            for y in ca:
                if not ok:
                    break
                if x[0] != y[0] or not (x[1] or y[1]):
                    continue
                lo, hi = max(x[2], y[2]), min(x[2] + x[3], y[2] + y[3])
                for pos in range(lo, hi):
                    conflict = True
                    if _stripe_of(pos - x[2], x[3], S) > _stripe_of(pos - y[2], y[3], S):
                        ok = False
                        break
        if conflict and ok:
            E[4] = 1

    vcS = [None] * nB

    def build_striped(all_non_aligned, with_vc):
        if with_vc:
            for b in range(nB):
                vcS[b] = [-1] * (S * nSt[b] * nB)
        ptr = [0] * nB
        done = 0
        progress = True
        while progress:
            progress = False
            for b in range(nB):
                nb = nSt[b]
                while ptr[b] < S * nb:
                    pos = ptr[b]
                    s, i = divmod(pos, nb)
                    ready = True
                    for e in inE[b][i]:
                        E = edges[e]
                        if E[4] and not all_non_aligned:
                            need = s * nSt[E[2]] + E[3]
                        else:
                            need = (S - 1) * nSt[E[2]] + E[3] if s == 0 else -1
                        if need >= 0 and ptr[E[2]] <= need:
                            ready = False
                            break
                    if not ready:
                        break
                    if with_vc:
                        v = vcS[b]
                        o = pos * nB
                        if pos > 0:
                            v[o:o + nB] = v[o - nB:o]
                        for e in inE[b][i]:
                            E = edges[e]
                            if E[4] and not all_non_aligned:
                                need = s * nSt[E[2]] + E[3]
                            else:
                                need = (S - 1) * nSt[E[2]] + E[3] if s == 0 else -1
                            if need < 0:
                                continue
                            u = vcS[E[2]]
                            uo = need * nB
                            for x in range(nB):
                                if u[uo + x] > v[o + x]:
                                    v[o + x] = u[uo + x]
                        v[o + b] = pos
                    ptr[b] += 1
                    done += 1
                    progress = True
        return done == evU * S

    res["cycle_step"] = 0 if build_striped(True, False) else 1
    pair_work = 0
    rnd = 0
    skipped = None
    while True:
        if rnd > len(edges) + 1:
            skipped = "demotion did not converge"
            break
        if not build_striped(False, True):
            res["cycle"] = 1
            skipped = "striped flag graph has a cycle (certain deadlock)"
            break
        n_demote, intra, stuck = 0, 0, False
        demoted_tb = [0] * nB
        g0, nrec = 0, len(recs)
        while g0 < nrec and skipped is None:
            g1 = g0
            key0 = recs[g0][0]
            while g1 < nrec and recs[g1][0] == key0:
                g1 += 1
            for x in range(g0, g1):
                X = recs[x]
                for y in range(x + 1, g1):
                    Y = recs[y]
                    if not (X[5] or Y[5]):
                        continue
                    if X[1] == Y[1] and X[2] == Y[2]:
                        continue
                    pair_work += 1
                    if pair_work > 256 * 1024 * 1024:
                        skipped = "pair budget"
                        break
                    if X[1] == Y[1]:
                        A, B = (X, Y) if X[2] < Y[2] else (Y, X)
                    elif hbU(X[1], X[2], Y[1], Y[2]):
                        A, B = X, Y
                    elif hbU(Y[1], Y[2], X[1], X[2]):
                        A, B = Y, X
                    else:
                        continue
                    kA, kB = _stripe_of(A[3], A[4], S), _stripe_of(B[3], B[4], S)
                    if A[1] == B[1]:
                        if kA > kB:
                            intra += 1
                        continue
                    posA, posB = kA * nSt[A[1]] + A[2], kB * nSt[B[1]] + B[2]
                    if vcS[B[1]][posB * nB + A[1]] >= posA:
                        continue
                    if (n_demote > 0) if rnd < 256 else demoted_tb[B[1]]:
                        continue
                    before = n_demote
                    for ps in (0, 1):
                        if n_demote != before:
                            break
                        ci = B[2]
                        while ci >= 0 and n_demote == before:
                            for e in inE[B[1]][ci]:
                                E = edges[e]
                                if not E[4]:
                                    continue
                                if ps == 0 and E[2] != A[1]:
                                    continue
                                if (E[3] >= A[2]) if E[2] == A[1] else hbU(A[1], A[2], E[2], E[3]):
                                    E[4] = 0
                                    n_demote += 1
                                    demoted_tb[B[1]] = 1
                                    break
                            ci -= 1
                    if n_demote == before:
                        stuck = True
                if skipped:
                    break
            g0 = g1
        if skipped:
            break
        res["intra"] = intra
        res["demoted"] += n_demote
        if n_demote == 0 and stuck:
            skipped = "violation without a demotable edge"
            break
        if n_demote == 0:
            break
        rnd += 1
    if skipped:
        res["skipped"] = skipped
        for E in edges:
            E[4] = 0
        res["cycle"] = res["cycle_step"]
    res["aligned"] = {(E[0], E[1]) for E in edges if E[4]}
    return res


def _topo_ok(nSt, inE, edges):
    nB = len(nSt)
    ptr = [0] * nB
    progress = True
    while progress:
        progress = False
        for b in range(nB):
            while ptr[b] < nSt[b]:
                if any(ptr[edges[e][2]] <= edges[e][3] for e in inE[b][ptr[b]]):
                    break
                ptr[b] += 1
                progress = True
    return ptr == nSt


def _striped_cycle(nSt, inE, edges, S):
    """True if the striped flag graph with every edge non-aligned (runtime_v2b's order) has a cycle."""
    nB = len(nSt)
    ptr = [0] * nB
    progress = True
    while progress:
        progress = False
        for b in range(nB):
            nb = nSt[b]
            while ptr[b] < S * nb:
                s, i = divmod(ptr[b], nb)
                if s == 0 and any(ptr[edges[e][2]] <= (S - 1) * nSt[edges[e][2]] + edges[e][3]
                                  for e in inE[b][i]):
                    break
                ptr[b] += 1
                progress = True
    return any(ptr[b] < S * nSt[b] for b in range(nB))



def rank_sig(tbs):
    """Key for caching the analysis: ranks with equal steps have equal results."""
    return tuple(tuple(tb["steps"]) for tb in tbs)



def stripe_flags(tbs, stripes, mode="auto"):
    """Flag choice of one rank for a launch with `stripes` stripes (rt3 host.cc:886-895).
    mode = NCCL_MSCCL_STRIPE_FLAGS: "step" (runtime_v2b), "stripe", or "auto" (runtime_v3 default).
    Returns (per-stripe flags used, aligned): aligned = {(TB index, step index)} of the consumer steps whose
    dependency waits at every stripe for the producer's same stripe; empty when step flags are used."""
    if mode == "step" or stripes <= 1:
        return False, frozenset()
    res = stripe_analyze(tbs, stripes, auto=(mode == "auto"))
    use = True if mode == "stripe" else bool(res["cycle_step"] and not res["cycle"])
    return use, (frozenset(res["aligned"]) if use else frozenset())
