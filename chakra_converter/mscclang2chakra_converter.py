#!/usr/bin/env python3

import logging

from  xml.etree import ElementTree
from typing import Any, List
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    NodeType,
    Node,
    AttributeProto as ChakraAttr,
    COMP_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    GlobalMetadata
)
import msccl_engine_model

HARDCODE_LOCAL_BW = 50
# HARDCODED. Refer to PacketBundle::call.
# 1000 b/c microsecond to nanosecond. Refer to Workload::issue_replay
def calculate_comp_time(data_size): 
    return int(3 * int(data_size / HARDCODE_LOCAL_BW) / 1000)

# M1 net_send_handoff defaults: runtime source counts (torchcomms-msccl v2_28 runtime_v2b 82c2bc6, src/):
#   depth 8        include/device.h:24 NCCL_STEPS; net.cc:1285 maxDepth = min(NCCL_STEPS, ...), :1297 the send proxy
#                  posts at most done + maxDepth slots; device/prims_simple.h:115, prims_ll.h:59 the GPU waits while
#                  NCCL_STEPS slots are outstanding; net.cc:1428-1429 / 1736-1742 slots return on network-send /
#                  receiver-consumption completion.
#   chunk_steps    include/msccl_engine.h:45 MSCCL_ENGINE_CHUNKSTEPS 4 (Simple), msccl_engine_host.cc:64-66 (1 for LL).
#   chunk_bytes    device/msccl_engine.cuh:219-221 chunkSize = calcBytePerStep x chunkSteps; device/primitives.h:34-36,
#                  49-51 calcBytePerStep = buffSize / NCCL_STEPS (LL: / 2, half is data); init.cc:807-809 default
#                  buffSizes LL 8 x 512 x 8 x 16 B = 512 KiB, Simple 4 MiB -> Simple 2 MiB, LL 32 KiB per chunk.
#   copy_GBps      absent (0 hand-off time). Measured per-block rates (Job 1: Simple copy 31.2, LL 8.2 GB/s) can be
#                  passed explicitly.
HANDOFF_DEFAULTS = dict(depth=8, chunk_steps={"Simple": 4, "LL": 1},
                        chunk_bytes={"Simple": 2 << 20, "LL": 32 << 10})

class MSCCLStep:    
    def add_parent(
        self,
        parent: "MSCCLStep",
    ) -> None: 
        if type(parent) is MSCCLNopStep:
            raise Exception(f"Nop parent added as parent")
        if type(self) is MSCCLNopStep:
            raise Exception(f"Trying to add parent to Nop")
        if type(parent) is MSCCLReceiveReduceComputeStep:
            self.node.data_deps.append(parent.comp_node.id)
            return
        self.node.data_deps.append(parent.node.id)
    
    def encode_message(
        self,
        g
    ) -> None:
        encode_message(g, self.node)
    
class MSCCLCompStep(MSCCLStep):
    def __init__(
        self,
        tb_xml_node: ElementTree.Element,
        step_id: int,
        node_id: int, 
        comp_data_size_bytes: int
    ) -> None:
        tb_id = tb_xml_node.attrib['id']

        node = Node()
        node.id = node_id
        node.name = f"COMP_NODE_tb{tb_id}_step{step_id}"
        node.type = COMP_NODE
        node.duration_micros = calculate_comp_time(comp_data_size_bytes)
        self.comp_ns = 3 * comp_data_size_bytes / HARDCODE_LOCAL_BW     # M1 comp_ns: the same rate, in ns
        self.node = node


class MSCCLSendStep(MSCCLStep):
    def __init__(
        self,
        tb_xml_node: ElementTree.Element,
        step_id: int, 
        node_id: int,
        comm_data_size_bytes: int,
        protocol: str,
        comm_delay_ns: int
    ) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.dst = int(tb_xml_node.attrib['send'])
        self.tag = int(tb_xml_node.attrib['chan'])

        node = Node()
        node.id = node_id
        node.name = f'COMM_SEND_NODE_tb{tb_id}_step{step_id}'
        node.type = COMM_SEND_NODE
        node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_SEND_NODE))
        node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=comm_data_size_bytes))
        node.attr.append(ChakraAttr(name="comm_dst",
                                    uint64_val=self.dst))
        node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=self.tag))
        node.attr.append(ChakraAttr(name="comm_protocol",
                                    string_val=protocol))
        node.attr.append(ChakraAttr(name="comm_delay_ns",
                                    uint64_val=comm_delay_ns))
        self.node = node

class MSCCLReceiveStep(MSCCLStep):
    def __init__(
        self,
        tb_xml_node: ElementTree.Element,
        step_id: int, 
        node_id: int,
        comm_data_size_bytes: int,
        protocol: str
    ) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.src = int(tb_xml_node.attrib['recv'])
        self.tag = int(tb_xml_node.attrib['chan'])

        node = Node()
        node.id = node_id
        node.name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        node.type = COMM_RECV_NODE
        node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_RECV_NODE))
        node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=comm_data_size_bytes))
        node.attr.append(ChakraAttr(name="comm_src",
                                    uint64_val=self.src))
        node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=self.tag))
        node.attr.append(ChakraAttr(name="comm_protocol",
                                    string_val=protocol))
        node.attr.append(ChakraAttr(name="comm_delay_ns",
                                    uint64_val=0))
        self.node = node

class MSCCLReceiveReduceComputeStep(MSCCLStep):
    def __init__(
        self,
        tb_xml_node: ElementTree.Element,
        step_id: int, 
        recv_node_id: int,
        comp_node_id: int,
        comm_data_size_bytes: int,
        comp_data_size_bytes: int,
        protocol: str
    ) -> None:
        tb_id = tb_xml_node.attrib['id']
        self.src = int(tb_xml_node.attrib['recv'])
        self.tag = int(tb_xml_node.attrib['chan'])

        recv_node = Node()
        recv_node.id = recv_node_id
        recv_node.name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        recv_node.type = COMM_RECV_NODE
        recv_node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_RECV_NODE))
        recv_node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=comm_data_size_bytes))
        recv_node.attr.append(ChakraAttr(name="comm_src",
                                    uint64_val=self.src))
        recv_node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=self.tag))
        recv_node.attr.append(ChakraAttr(name="comm_protocol",
                                    string_val=protocol))
        recv_node.attr.append(ChakraAttr(name="comm_delay_ns",
                                    uint64_val=0))
        self.recv_node = recv_node

        comp_node = Node()
        comp_node.id = comp_node_id
        comp_node.name = f"COMP_NODE_tb{tb_id}_step{step_id}"
        comp_node.type = COMP_NODE
        comp_node.duration_micros = calculate_comp_time(comp_data_size_bytes)
        self.comp_ns = 3 * comp_data_size_bytes / HARDCODE_LOCAL_BW     # M1 comp_ns: the same rate, in ns
        comp_node.data_deps.append(recv_node.id)
        self.comp_node = comp_node

    def encode_message(
        self,
        g
    ) -> None:
        encode_message(g, self.recv_node)
        encode_message(g, self.comp_node)

    def add_parent(
        self,
        parent: "MSCCLStep",
    ) -> None: 
        if type(parent) is MSCCLReceiveReduceComputeStep:
            self.recv_node.data_deps.append(parent.comp_node.id)
            return
        self.recv_node.data_deps.append(parent.node.id)

class MSCCLNopStep(MSCCLStep):
    def __init__(
        self,
    ) -> None:
        self.name = f"NOP_Node"

class MSCCL2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        coll_size: int,
        ll_size_factor: float,
        simple_latency_factor: float,
        gpus_per_node: int,
        intra_latency_ns: float,
        inter_latency_ns: float,
        protocol_override: str | None,
        logger: logging.Logger,
        msccl_stripe_calls: bool = False,
        msccl_conn_tags: bool = False,
        net_hop_ns: dict | None = None,
        nvl_hop_ns: dict | None = None,
        call_cost_ns: dict | None = None,
        ll_net_size_factor: float | None = None,
        ll_conn_GBps: float | None = None,
        net_link_GBps: float = 23.3,
        proxy_chan_ns: dict | None = None,
        proxy_chan_scope: str = "rank",
        stage4: dict | None = None,
        comp_ns: bool = False,
        yaml_bw_unit: str = "GiB/s",
        nic_direct: bool = False,
        net_send_handoff: dict | bool | None = None,
        stage5: dict | None = None,
        msccl_stripe_runtime: str = "v2b"
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.logger = logger
        self.next_node_id = 0
        self.collective_size = coll_size #Bytes
        self.ll_size_factor = ll_size_factor
        self.simple_latency_factor = simple_latency_factor
        self.gpus_per_node = gpus_per_node
        self.intra_latency_ns = intra_latency_ns
        self.inter_latency_ns = inter_latency_ns
        self.protocol_override = protocol_override
        # MSCCL engine model, see msccl_engine_model.py. Both default off: the ETs are then
        # byte-identical to the converter without these options.
        self.msccl_stripe_calls = msccl_stripe_calls
        self.msccl_conn_tags = msccl_conn_tags
        # Stripe rules of the target runtime (msccl_engine_model.py, parts 3 and 4). "v2b" (default): nStripes per
        # rank from the AddAlgo formula, unchanged ETs. "v3" (runtime_v3 defaults): nStripes per launch (R4a), and
        # with stage5 stripe_passes the per-stripe dependency flags of NCCL_MSCCL_STRIPE_FLAGS=auto (R4b).
        if msccl_stripe_runtime not in ("v2b", "v3"):
            raise ValueError(f"msccl_stripe_runtime must be 'v2b' or 'v3', not {msccl_stripe_runtime!r}")
        self.msccl_stripe_runtime = msccl_stripe_runtime
        self._launch_stripes = None   # runtime_v3: nStripes of this launch, the same on every rank (convert())
        self._v3_info = dict(stripe_flag_ranks=0, aligned_deps=0)   # runtime_v3 flags: ranks using them, marks
        # Stage 3 measured terms (2-node micro, job 60948477; Job 1). All default off (None):
        # with every one off the ETs are byte-identical to Stage 1 (9262e7c). See STAGE3 below.
        self.net_hop_ns = net_hop_ns
        self.nvl_hop_ns = nvl_hop_ns
        self.call_cost_ns = call_cost_ns
        self.ll_net_size_factor = ll_net_size_factor
        self.ll_conn_GBps = ll_conn_GBps
        self.net_link_GBps = net_link_GBps
        self.proxy_chan_ns = proxy_chan_ns
        if proxy_chan_scope not in ("rank", "peer"):
            raise ValueError(f"proxy_chan_scope must be 'rank' or 'peer', not {proxy_chan_scope!r}")
        self.proxy_chan_scope = proxy_chan_scope
        self.stage4 = stage4
        # M1 plain-defect fixes (calib/c20261105/m1_fixes/REPORT.md). All default off: the ETs are then
        # byte-identical to 0886102. See M1 below.
        self.comp_ns = comp_ns
        if yaml_bw_unit not in ("GiB/s", "GB/s"):
            raise ValueError(f"yaml_bw_unit must be 'GiB/s' or 'GB/s', not {yaml_bw_unit!r}")
        self.yaml_bw_unit = yaml_bw_unit
        self.nic_direct = nic_direct
        if net_send_handoff is True:
            net_send_handoff = {}
        self.net_send_handoff = None if not isinstance(net_send_handoff, dict) else dict(HANDOFF_DEFAULTS, **net_send_handoff)
        # Stage 5 mechanism model (calib/c20261105/stage5/SPEC.md). Absent: the ETs are byte-identical to 8335a8c.
        self.stage5 = stage5
        self._s5_stripes = None   # Stage 5b iters: nStripes per rank of the original IR (None: computed per rank)
        self._s5_iter_g = 1       # Stage 5b iters: iterations per unrolled copy
        self._s5_zcalls = None    # Stage 5b stripe_passes: {(gpu, tb, step): [empty calls before, after]}
        self._s5_passfirst = set()  # Stage 5c: (gpu, tb, step) of a TB's first call in a stripe pass k >= 1
        self._s5_chunk_B = 0      # bytes per MSCCL chunk of the original IR (maxAllowedCount)
        if stage5 is not None:
            clash = [k for k, v in (("stage4", stage4), ("net_hop_ns", net_hop_ns), ("nvl_hop_ns", nvl_hop_ns),
                                    ("call_cost_ns", call_cost_ns), ("ll_net_size_factor", ll_net_size_factor),
                                    ("ll_conn_GBps", ll_conn_GBps), ("proxy_chan_ns", proxy_chan_ns),
                                    ("net_send_handoff", net_send_handoff)) if v is not None]
            if clash:
                raise ValueError(f"stage5 replaces {clash}: pass stage5 alone")
            if not (comp_ns and yaml_bw_unit == "GB/s" and nic_direct and msccl_stripe_calls and msccl_conn_tags):
                raise ValueError("stage5 needs comp_ns, yaml_bw_unit='GB/s', nic_direct, msccl_stripe_calls and "
                                 "msccl_conn_tags (M1 fixes 1, 2, 5 and the engine model)")
        for name, latency_ns, hop in (("net_hop_ns", 2 * inter_latency_ns, net_hop_ns),
                                      ("nvl_hop_ns", intra_latency_ns, nvl_hop_ns)):
            for proto, hop_ns in (hop or {}).items():
                if hop_ns < latency_ns:
                    raise ValueError(f"{name}[{proto}] = {hop_ns} ns is below the route's link latency "
                                     f"{latency_ns} ns: lower the network YAML latency (and the matching "
                                     f"intra/inter_latency_ns argument) instead")
        print('collective_size', self.collective_size)

    def is_network(self, src_rank: int, dst_rank: int) -> bool:
        return src_rank // self.gpus_per_node != dst_rank // self.gpus_per_node

    def effective_comm_size(self, protocol: str, payload_size_bytes: int,
                            src_rank: int = -1, dst_rank: int = -1) -> int:
        if protocol == "LL":
            if self.ll_net_size_factor is not None and self.is_network(src_rank, dst_rank):
                # T3 (per-NIC cap): an LL network message occupies the NIC as if it carried
                # ll_net_size_factor x its payload, so the NIC moves at most link/factor of LL data.
                return int(payload_size_bytes * self.ll_net_size_factor)
            return int(payload_size_bytes * self.ll_size_factor)
        return payload_size_bytes

    def protocol_send_delay_ns(self, protocol: str, src_rank: int, dst_rank: int) -> int:
        # T1 / T5: measured per-hop cost per protocol. The route already pays its YAML link
        # latencies (2 network links, or 1 NVLink link); the send delay carries the rest.
        if self.net_hop_ns is not None and self.is_network(src_rank, dst_rank):
            return int(self.net_hop_ns[protocol] - 2 * self.inter_latency_ns)
        if self.nvl_hop_ns is not None and not self.is_network(src_rank, dst_rank):
            return int(self.nvl_hop_ns[protocol] - self.intra_latency_ns)
        if protocol != "Simple":
            return 0

        link_latency_ns = self.intra_latency_ns
        if src_rank // self.gpus_per_node != dst_rank // self.gpus_per_node:
            link_latency_ns = self.inter_latency_ns

        extra_latency_ns = max(0.0, self.simple_latency_factor - 1.0) * link_latency_ns
        return int(extra_latency_ns)

    def route_latency_ns(self, src_rank: int, dst_rank: int) -> float:
        """Link latency of the route the analytical backend takes on the 2-dim network
        [FullyConnected GPUs, Switch nodes]: one intra-node link inside a node; between nodes
        two switch links, plus one intra-node link when the local GPU index differs
        (dimension-ordered routing)."""
        if src_rank // self.gpus_per_node == dst_rank // self.gpus_per_node:
            return self.intra_latency_ns
        latency_ns = 2 * self.inter_latency_ns
        if self.nic_direct:
            # M1 nic_direct: the backend runs the Switch dim as one global switch (switch_scope global):
            # GPU i -> own NIC link -> switch -> GPU j's own NIC link, no NVLink leg
            return latency_ns
        if src_rank % self.gpus_per_node != dst_rank % self.gpus_per_node:
            latency_ns += self.intra_latency_ns
        return latency_ns

    @staticmethod
    def _attr(node: Any, name: str) -> Any:
        for attr in node.attr:
            if attr.name == name:
                return attr
        raise KeyError(f"node {node.id} has no attribute {name}")

    def _n_stripes(self, gpu: Any, protocol: str) -> int:
        """nStripes of one rank: runtime_v2b's per-rank AddAlgo value, or runtime_v3's per-launch value."""
        if self._launch_stripes is not None:
            return self._launch_stripes
        return msccl_engine_model.n_stripes(gpu, protocol)

    def _set_launch_stripes(self, root: Any, protocol: str) -> None:
        """msccl_stripe_runtime "v3": the launch stripe count of the IR as given (before any unroll), from the
        message size (msccl_engine_host.cc:863-881 of runtime_v3). AllReduce only: the runtime has no MSCCL
        launch for other collectives, so they keep the AddAlgo formula."""
        if self.msccl_stripe_runtime != "v3" or root.attrib.get("coll", "allreduce") != "allreduce":
            return
        gpus = root.findall('gpu')
        ncpl = int(root.attrib.get('nchunksperloop') or 0) or \
            max(int(gp.attrib['i_chunks']) or int(gp.attrib['o_chunks']) for gp in gpus)
        self._launch_stripes = msccl_engine_model.launch_stripes(
            (msccl_engine_model.rank_tbs(gp) for gp in gpus), protocol, int(self.collective_size), ncpl)

    def apply_stripe_calls(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """A send that stands for m stripe calls pays (m - 1) more per-message costs
        (msccl_engine_model.py, part 1)."""
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            stripes = self._n_stripes(gpu, protocol)
            if stripes <= 1:
                continue
            for tb_id, steps in node_map[gpu_id].items():
                for step_id, et_node in steps.items():
                    step = step_map[gpu_id][tb_id][step_id]
                    if step.attrib['type'] != "s":
                        continue
                    calls = msccl_engine_model.stripe_calls(int(step.attrib['cnt']), stripes)
                    if calls <= 1:
                        continue
                    delay = self._attr(et_node.node, "comm_delay_ns")
                    message_cost_ns = self.route_latency_ns(gpu_id, et_node.dst) + delay.uint64_val
                    delay.uint64_val = int(delay.uint64_val + (calls - 1) * message_cost_ns)

    def apply_conn_tags(self, root: Any, node_map: dict, step_map: dict) -> None:
        """tag = channel + 64 * connIndex (msccl_engine_model.py, part 2)."""
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            slots = msccl_engine_model.conn_index(gpu)
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                recv_slot, send_slot = slots[tb_id]
                if recv_slot == 0 and send_slot == 0:
                    continue
                chan = int(tb.attrib['chan'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    step_type = step_map[gpu_id][tb_id][step_id].attrib['type']
                    if step_type == "s" and send_slot:
                        et_node.tag = msccl_engine_model.conn_tag(chan, send_slot)
                        self._attr(et_node.node, "comm_tag").uint64_val = et_node.tag
                    elif step_type == "r" and recv_slot:
                        et_node.tag = msccl_engine_model.conn_tag(chan, recv_slot)
                        self._attr(et_node.node, "comm_tag").uint64_val = et_node.tag
                    elif step_type == "rrc" and recv_slot:
                        et_node.tag = msccl_engine_model.conn_tag(chan, recv_slot)
                        self._attr(et_node.recv_node, "comm_tag").uint64_val = et_node.tag

    # STAGE3 ---------------------------------------------------------------------------------
    # Measured terms (coll_evolve_runs/calib/c20261105/profiling/RESULTS_micro_60948477.md, Job 1
    # job1/60919901/RESULTS_running.md). Order inside convert(): T1/T5 enter the send delay at
    # node creation and T4 before the stripe calls, so a send that stands for m stripe calls pays
    # them m times (they are per-message costs). T3 (a per-byte time) and T2 (once per call) are
    # added after the stripe calls and are never multiplied.

    def _network_channels(self, gpu: Any) -> int:
        """Channels of a rank with at least one network connection in use (send or receive)."""
        chans = set()
        gpu_id = int(gpu.attrib['id'])
        for tb in gpu.findall('tb'):
            types = {s.attrib['type'] for s in tb.findall('step')}
            send, recv = int(tb.attrib['send']), int(tb.attrib['recv'])
            uses_net = (send >= 0 and types & msccl_engine_model.SEND_TYPES and self.is_network(gpu_id, send)) or \
                       (recv >= 0 and types & msccl_engine_model.RECV_TYPES and self.is_network(recv, gpu_id))
            if uses_net:
                chans.add(int(tb.attrib['chan']))
        return len(chans)

    def _send_channels_by_peer(self, gpu: Any) -> dict:
        """{network peer: set of channels of this rank with a TB that sends to that peer}."""
        gpu_id = int(gpu.attrib['id'])
        out = {}
        for tb in gpu.findall('tb'):
            send = int(tb.attrib['send'])
            if send < 0 or not self.is_network(gpu_id, send):
                continue
            if any(s.attrib['type'] in msccl_engine_model.SEND_TYPES for s in tb.findall('step')):
                out.setdefault(send, set()).add(int(tb.attrib['chan']))
        return out

    def apply_proxy_chan(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """T4: every network send pays alpha x (number of OTHER network channels of its rank).
        proxy_chan_scope="peer" (T4'): only the other channels of the rank that send to the SAME
        network peer count (the condition of the channel-axis micro: c channels, one peer pair)."""
        alpha_ns = self.proxy_chan_ns[protocol]
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            if self.proxy_chan_scope == "peer":
                by_peer = self._send_channels_by_peer(gpu)
                others_of = lambda dst: len(by_peer.get(dst, ())) - 1  # noqa: E731
            else:
                n_rank = self._network_channels(gpu) - 1
                others_of = lambda dst: n_rank  # noqa: E731
            for tb_id, steps in node_map[gpu_id].items():
                for step_id, et_node in steps.items():
                    if step_map[gpu_id][tb_id][step_id].attrib['type'] != "s":
                        continue
                    if not self.is_network(gpu_id, et_node.dst):
                        continue
                    others = others_of(et_node.dst)
                    if others <= 0:
                        continue
                    delay = self._attr(et_node.node, "comm_delay_ns")
                    delay.uint64_val = int(delay.uint64_val + others * alpha_ns)

    def apply_ll_conn_rate(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """T3 (per-connection rate): an LL network send moves its payload at most at ll_conn_GBps.
        A connection (rank, chan, peer, dir, connIndex) is one TB's send side (conn tags), so its
        sends are already serialized by the TB's control edges. The message's own unshared
        transfer time (comm_size / link) is subtracted, so one connection alone runs at
        ll_conn_GBps and not at ll_conn_GBps in series with the NIC cap."""
        if protocol != "LL":
            return
        link_Bpns = self.net_link_GBps * self._yaml_Bpns_per_unit()   # AstraSim reads GB/s as GiB/s (unless M1 unit)
        conn_Bpns = self.ll_conn_GBps                           # measured GB/s = B/ns
        head = (self.stage4 or {}).get("ll_conn_head")          # Stage 4b LLR' (absent: one rate)
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            for tb_id, steps in node_map[gpu_id].items():
                for step_id, et_node in steps.items():
                    step = step_map[gpu_id][tb_id][step_id]
                    if step.attrib['type'] != "s" or not self.is_network(gpu_id, et_node.dst):
                        continue
                    payload = et_node.payload_bytes
                    wire = self._attr(et_node.node, "comm_size").uint64_val
                    if head:
                        first = min(payload, head["bytes"])
                        extra_ns = first / head["GBps"] + (payload - first) / conn_Bpns - wire / link_Bpns
                    else:
                        extra_ns = payload / conn_Bpns - wire / link_Bpns
                    if extra_ns > 0:
                        delay = self._attr(et_node.node, "comm_delay_ns")
                        delay.uint64_val = int(delay.uint64_val + extra_ns)

    # STAGE4 ---------------------------------------------------------------------------------
    # Structural per-rank proxy model + per-step kernel costs (PREREG_stage4.md). Needs the AstraSim
    # branch krydzy/stage4-proxy (node attributes proxy_net, proxy_svc_ns, proxy_inject_ns,
    # proxy_poll_ns, comm_post_delay_ns, receive comm_delay_ns). Keys of `stage4`:
    #   mode            "poll": receive-side cost poll x (other active network ops), dynamic
    #                   "fifo": the rank's proxy is a FIFO server (send s_send, receive s_recv)
    #                   "none": the same fixed costs, no proxy interaction (drop-one of the proxy)
    #   net_hop_ns      {proto: ns} one network hop with one op, sender step end -> receiver step end
    #   s_send_ns       proxy send pickup; s_recv_ns {proto: ns} receive flush/test
    #   poll_ns         {proto: ns} (poll mode)
    #   step_ns         step work per step; xtb_ns per cross-TB dependency
    #   send_work_ns    {proto: {"net": ns, "nvl": ns}} GPU work of a send step
    #   nvl_hop_ns      {proto: ns} one NVLink hop incl. the send step (sender step start -> receiver step end)
    #   call_ns         {proto: ns} per call (outside kernel + setup), on every root send
    #   reduce_GBps     {proto: GB/s} rrc compute rate
    #   ll_scan_GBps, ll_conn_GBps, net_link_GBps   LL network rate (fifo: proxy scan occupancy + per-connection
    #                   injection; poll: use the Stage 3 T3 options instead)
    # Route latencies come from the YAML: intra_latency_ns / inter_latency_ns must equal it.
    # Stage 4b keys (calib/c20261105/stage4b/PREREG_stage4b.md), all optional; absent = 6741972 ETs byte for byte.
    # Need astra-sim krydzy/stage4b for xtb_mode / recv_work_ns (node attributes xtb_parent, xtb_bind_ns,
    # xtb_chk_ns, recv_work_ns); the others only change existing attributes.
    #   xtb_mode "edge" K': a step with a cross-TB depid starts at max(own previous step + xtb_chk_ns[proto],
    #                   cross-TB producer + xtb_ns), once per step (not x stripe calls). xtb leaves the send
    #                   pre-delay and the receive posting delay.
    #   zero_stripe_ns  Z: (nStripes - m) x zero_stripe_ns per step on a striped rank (stripe calls that move
    #                   no data): send pre-delay (after the stripe calls), receive post delay
    #   recv_work_ns    W: {proto: ns} receive finish >= posting time + m x W (+ Z of the step)
    #   rt_table        RT: {proto: [[tbs, ns], ...]} receive-side cost per network receive (x m), linear in the
    #                   receiving rank's TB count between the points, flat outside; added to the post delay
    #   ll_conn_head    LLR': {"bytes": B, "GBps": r}: an LL network send moves its first B payload bytes at r and
    #                   the rest at ll_conn_GBps (the per-connection term of apply_ll_conn_rate)

    def _set_attr(self, node: Any, name: str, value: int) -> None:
        for attr in node.attr:
            if attr.name == name:
                attr.uint64_val = int(value)
                return
        node.attr.append(ChakraAttr(name=name, uint64_val=int(value)))

    def _recv_node(self, et_node: Any) -> Any:
        return et_node.recv_node if type(et_node) is MSCCLReceiveReduceComputeStep else et_node.node

    def apply_stage4_pre(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """Per-message costs, before the stripe calls (so a send of m stripe calls pays them m times)."""
        c = self.stage4
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    step = step_map[gpu_id][tb_id][step_id]
                    kind = step.attrib['type']
                    xtb = c["xtb_ns"] if int(step.attrib['depid']) not in (-1, tb_id) else 0
                    if xtb and c.get("xtb_mode") == "edge":
                        self._xtb_edge(gpu_id, tb_id, step, et_node, kind, node_map, protocol)
                        xtb = 0
                    if kind == "s":
                        dst = int(tb.attrib['send'])
                        net = self.is_network(gpu_id, dst)
                        work = c["send_work_ns"][protocol]["net" if net else "nvl"]
                        if net:
                            # remainder of the one-op hop not carried by the route latency, pickup,
                            # flush and the receiver's step work
                            rest = c["net_hop_ns"][protocol] - self.route_latency_ns(gpu_id, dst) \
                                - c["s_send_ns"] - c["s_recv_ns"][protocol] - c["step_ns"]
                            if gpu_id % self.gpus_per_node != dst % self.gpus_per_node and not self.nic_direct:
                                rest += self.intra_latency_ns      # mixed-index route keeps its NVLink leg
                        else:
                            rest = c["nvl_hop_ns"][protocol] - work - self.intra_latency_ns - c["step_ns"]
                        delay = work + xtb + max(0.0, rest) + (c["s_send_ns"] if net and c["mode"] != "fifo" else 0)
                        self._attr(et_node.node, "comm_delay_ns").uint64_val = int(delay)
                        if net:
                            self._set_attr(et_node.node, "proxy_net", 1)
                    elif kind in ("r", "rrc"):
                        rn = self._recv_node(et_node)
                        src = int(tb.attrib['recv'])
                        if xtb:
                            self._attr(rn, "comm_delay_ns").uint64_val = int(xtb)
                        if self.is_network(src, gpu_id):
                            self._set_attr(rn, "proxy_net", 1)
                        if kind == "rrc":
                            rate = c["reduce_GBps"][protocol]
                            et_node.comp_node.duration_micros = int(et_node.payload_bytes / rate / 1e3)
                            et_node.comp_ns = et_node.payload_bytes / rate      # M1 comp_ns

    def _xtb_edge(self, gpu_id: int, tb_id: int, step: Any, et_node: Any, kind: str, node_map: dict,
                  protocol: str) -> None:
        """K': mark the step's ET node (send node, or the receive node of r / rrc) with its cross-TB parent."""
        c = self.stage4
        if kind not in ("s", "r", "rrc"):
            return
        dep = node_map[gpu_id][int(step.attrib['depid'])][int(step.attrib['deps'])]
        if type(dep) is MSCCLNopStep:
            return                      # add_parent() refuses a nop parent: no cross-TB edge in the ET
        parent_id = dep.comp_node.id if type(dep) is MSCCLReceiveReduceComputeStep else dep.node.id
        chk, bind = c["xtb_chk_ns"][protocol], c["xtb_ns"]
        if chk > bind:
            raise ValueError(f"xtb_chk_ns {chk} > xtb_ns {bind}: AstraSim's xtb_edge_delay needs chk <= bind")
        n = et_node.node if kind == "s" else self._recv_node(et_node)
        self._set_attr(n, "xtb_parent", parent_id)
        self._set_attr(n, "xtb_bind_ns", bind)
        self._set_attr(n, "xtb_chk_ns", chk)

    @staticmethod
    def _interp(table: list, x: float) -> float:
        pts = sorted(table)
        if x <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    def apply_stage4b_post(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """Z, W, RT (after apply_stage4_post, so they never pass through the stripe-call multiplication)."""
        c = self.stage4
        z_ns = c.get("zero_stripe_ns")
        w_ns = (c.get("recv_work_ns") or {}).get(protocol)
        rt = (c.get("rt_table") or {}).get(protocol)
        if not (z_ns or w_ns or rt):
            return
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            stripes = self._n_stripes(gpu, protocol) if self.msccl_stripe_calls else 1
            n_tbs = len(gpu.findall('tb'))
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    kind = step_map[gpu_id][tb_id][step_id].attrib['type']
                    if kind not in ("s", "r", "rrc"):
                        continue
                    cnt = int(step_map[gpu_id][tb_id][step_id].attrib['cnt'])
                    m = max(1, msccl_engine_model.stripe_calls(cnt, stripes)) if stripes > 1 else 1
                    z = (stripes - m) * z_ns if z_ns and stripes > 1 else 0
                    if kind == "s":
                        if z:
                            d = self._attr(et_node.node, "comm_delay_ns")
                            d.uint64_val = int(d.uint64_val + z)
                        continue
                    rn = self._recv_node(et_node)
                    post = z
                    if rt and self.is_network(int(tb.attrib['recv']), gpu_id):
                        post += m * self._interp(rt, n_tbs)
                    if post:
                        self._set_attr(rn, "comm_post_delay_ns", self._attr(rn, "comm_post_delay_ns").uint64_val + int(post))
                    if w_ns:
                        self._set_attr(rn, "recv_work_ns", int(m * w_ns + z))

    def apply_stage4_post(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        """Receive-side costs (x stripe calls), proxy occupancies, LL rate (fifo), per-call cost on roots."""
        c = self.stage4
        link_Bpns = c.get("net_link_GBps", 23.3) * self._yaml_Bpns_per_unit()
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            stripes = self._n_stripes(gpu, protocol) if self.msccl_stripe_calls else 1
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    step = step_map[gpu_id][tb_id][step_id]
                    kind = step.attrib['type']
                    m = max(1, msccl_engine_model.stripe_calls(int(step.attrib['cnt']), stripes)) if stripes > 1 else 1
                    if kind in ("r", "rrc"):
                        rn = self._recv_node(et_node)
                        net = self.is_network(int(tb.attrib['recv']), gpu_id)
                        post = c["step_ns"]
                        if net and c["mode"] != "fifo":
                            post += c["s_recv_ns"][protocol]
                        if net and c["mode"] == "poll":
                            self._set_attr(rn, "proxy_poll_ns", m * c["poll_ns"][protocol])
                        if net and c["mode"] == "fifo":
                            self._set_attr(rn, "proxy_svc_ns", m * c["s_recv_ns"][protocol])
                        self._set_attr(rn, "comm_post_delay_ns", m * post)
                    elif kind == "s" and c["mode"] == "fifo" and self.is_network(gpu_id, et_node.dst):
                        svc = m * c["s_send_ns"]
                        inject = svc
                        if protocol == "LL":
                            P = et_node.payload_bytes
                            wire = self._attr(et_node.node, "comm_size").uint64_val
                            svc += P / c["ll_scan_GBps"]
                            inject = max(svc, m * c["s_send_ns"] + P / c["ll_conn_GBps"] - wire / link_Bpns)
                        self._set_attr(et_node.node, "proxy_svc_ns", max(1, svc))
                        self._set_attr(et_node.node, "proxy_inject_ns", inject)
            # per call: every root send of the rank
            for tb_id, steps in node_map[gpu_id].items():
                for et_node in steps.values():
                    if type(et_node) is MSCCLSendStep and len(et_node.node.data_deps) == 0:
                        d = self._attr(et_node.node, "comm_delay_ns")
                        d.uint64_val = int(d.uint64_val + c["call_ns"][protocol])

    def apply_call_cost(self, protocol: str, node_map: dict) -> None:
        """T2: per-call fixed cost at the start of every rank's program. Every send without a
        parent (a root of the rank's graph) waits call_cost_ns. A root receive needs no delay: it
        completes when its data arrives, and every send in the program starts after a root send."""
        cost_ns = self.call_cost_ns[protocol]
        for gpu_id in node_map:
            for tb_id, steps in node_map[gpu_id].items():
                for et_node in steps.values():
                    if type(et_node) is MSCCLSendStep and len(et_node.node.data_deps) == 0:
                        delay = self._attr(et_node.node, "comm_delay_ns")
                        delay.uint64_val = int(delay.uint64_val + cost_ns)

    # M1 -----------------------------------------------------------------------------------------
    # Plain-defect fixes (calib/c20261105/m1_fixes/REPORT.md), each opt-in:
    #   comp_ns          every COMP node also carries its exact compute time as attribute duration_ns (rounded to
    #                    1 ns); AstraSim krydzy/m1-fixes then uses it instead of duration_micros x 1000, which
    #                    truncates to whole us (0 below 16.7 KB at the legacy rate). Same rates, no new number.
    #   yaml_bw_unit     "GB/s": the network YAML carries bandwidth_unit GB/s (analytical krydzy/m1-fixes), so a
    #                    configured X means X B/ns; the converter's own link-rate arithmetic follows it.
    #   nic_direct       the network YAML carries switch_scope [local, global]: a cross-rail message goes
    #                    GPU i -> own NIC -> fabric -> GPU j's NIC (runtime_v2b N1 fix: each side of a NET
    #                    connection uses its own GPU's NIC, no PXN); route latency has no NVLink leg.
    #   net_send_handoff a network send finishes when its data is in the connection's slots (handed to the proxy),
    #                    not at arrival. Connection = (src, dst, tag) with depth NCCL_STEPS = 8 slots; a send step
    #                    of one stripe call takes ceil(cnt / nStripes) x chunkSteps slots; slots are freed when the
    #                    matching receive finishes. Eligible only when its data fits: one engine iteration (chunk
    #                    <= chunkSize) and slots <= depth; other network sends keep the arrival rule (slots 0).
    #                    Hand-off time after the slots are taken: payload / copy_GBps[proto] (0 if absent).
    #                    Needs AstraSim krydzy/m1-fixes (handoff_depth / handoff_slots / handoff_ns / handoff_release).

    def _yaml_Bpns_per_unit(self) -> float:
        return 1.0 if self.yaml_bw_unit == "GB/s" else (1 << 30) / 1e9

    def apply_comp_ns(self, node_map: dict) -> None:
        for gpu_id in node_map:
            for steps in node_map[gpu_id].values():
                for et_node in steps.values():
                    if type(et_node) is MSCCLReceiveReduceComputeStep:
                        self._set_attr(et_node.comp_node, "duration_ns", int(round(et_node.comp_ns)))
                    elif type(et_node) is MSCCLCompStep:
                        self._set_attr(et_node.node, "duration_ns", int(round(et_node.comp_ns)))

    def handoff_slots(self, gpu: Any, protocol: str, step: Any, chunk_size: int) -> int:
        """Slots one stripe call of this send step takes, or 0 when the send cannot be handed off whole."""
        h = self.net_send_handoff
        cnt = int(step.attrib['cnt'])
        stripes = self._n_stripes(gpu, protocol)
        per_call = -(-cnt // stripes)
        slots = per_call * h["chunk_steps"][protocol]
        if slots > h["depth"] or chunk_size > h["chunk_bytes"][protocol]:
            return 0
        return slots

    def apply_net_send_handoff(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        h = self.net_send_handoff
        copy = (h.get("copy_GBps") or {}).get(protocol)
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            num_chunks = int(gpu.attrib['i_chunks']) or int(gpu.attrib['o_chunks'])
            chunk_size = int(self.collective_size / num_chunks)
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    step = step_map[gpu_id][tb_id][step_id]
                    kind = step.attrib['type']
                    if kind == "s" and self.is_network(gpu_id, int(tb.attrib['send'])):
                        slots = self.handoff_slots(gpu, protocol, step, chunk_size)
                        self._set_attr(et_node.node, "handoff_depth", h["depth"])
                        self._set_attr(et_node.node, "handoff_slots", slots)
                        if slots and copy:
                            self._set_attr(et_node.node, "handoff_ns", int(round(et_node.payload_bytes / copy)))
                    elif kind in ("r", "rrc") and self.is_network(int(tb.attrib['recv']), gpu_id):
                        self._set_attr(self._recv_node(et_node), "handoff_release", 1)

    # STAGE5 ---------------------------------------------------------------------------------------
    # Mechanism model of the MSCCL runtime (calib/c20261105/stage5/SPEC.md, TERMS.md). Needs astra-sim krydzy/stage5
    # (node attributes s5_*), the M1 fixes (bandwidth unit GB/s, comp_ns, nic_direct) and the engine model. Every
    # value comes from the spec dict; the converter has no numbers of its own. Keys (times in ns, rates in GB/s = B/ns):
    #   placement   {"mode": "uniform", "class": "S"|"C"|"X"} | {"mode": "blocks", "nodes_per_chassis": n,
    #               "chassis_per_cabinet": m} | {"mode": "map", "nodes": [[cabinet, chassis], ...]}  (node = rank // 4)
    #   net         network message: "nic_GBps", "rdzv_threshold_B" (wire bytes above it: rendezvous, 3 traversals),
    #               "idc_limit_B", "hop_ns" {proto: {"idc"|"eager"|"rdzv": [[wire B, ns], ...]}} one-op relay hop at
    #               distance S (sender step start -> receiver step end, isolated call), "dist_ns" {proto: {"C"|"X": ns}}
    #               per fabric traversal over S, "rdzv_dist_ns" {proto: {"X": ns}} rendezvous extra over S (C: 2 x
    #               dist), "send_w0_ns" / "send_GBps" {proto} GPU send-step work, "tail_ns" {proto} receive step after
    #               the data is visible, "recv_flush_ns" {proto} receive latency after arrival (Simple GDR flush),
    #               "rdzv_occ_ns" {proto} (optional) per-connection occupancy of a rendezvous message
    #   proxy       (optional) one FIFO server per rank: "msg_ns" per network message (each side), "pre_frac" share
    #               booked at posting, "msgs_per_slot" {proto}, "slot_B" {proto}, "ll_scan_GBps" (LL sends),
    #               "simple_byte_GBps" (Simple, both sides); the Simple receive waits for its completion work
    #   conn        (optional) per-connection pipeline: {"Simple": {"GBps"}, "LL": {"head_B", "head_GBps", "GBps"}}
    #   handoff     (optional) {"depth", "chunk_steps" {proto}, "chunk_bytes" {proto}}; slots freed at the sender
    #   nvl         NVLink: "GBps", "hop_ns" {proto} (sender step end -> receiver step end), "tail_ns" {proto},
    #               "send_w0_ns" {proto}, "pace_GBps" {proto} (one TB's copy pace)
    #   rrc         (optional; absent: legacy compute) exposed reduce of the last FIFO slot:
    #               {proto: {"net"|"nvl": {"slot_B", "rrc_GBps", "copy_GBps"} | null}}
    #   call        (optional) {"gap_ns", "setup_ns" {proto}} on every root step of a rank
    #   xtb         (optional) K': {"bind_ns", "chk_ns" {proto}}
    #   zero_stripe_ns  (optional) {proto} Z per empty stripe call
    #   calls       (optional, default 1) back-to-back calls: K copies of the program per rank, copy j+1 starts
    #               after every step of copy j on that rank (kernel order on one stream) + the call cost
    # Stage 5b keys (calib/c20261105/stage5b/SPEC.md; each optional, absent = Stage 5 ETs byte for byte):
    #   placement   + {"mode": "uniform", "class": "N"|"F"} (half-cabinet near / far) and {"mode": "map", "rule":
    #               "half_cabinet", "nodes": [[cabinet, chassis], ...]}: near = same cabinet and same chassis half
    #               (chassis 0-3 / 4-7), far = the rest. "N" pays no distance term (as "S"); "F" uses
    #               net.dist_ns[proto]["F"] and net.rdzv_dist_ns[proto]["F"]
    #   ll_rank     {"GBps": r, "extra_conn_ns": x (optional)}: every LL network send of a rank shares one per-rank
    #               data server (s5_rank_ns = payload / r); x is added once per extra LL network connection of the
    #               rank (sensitivity arm). Use it with no conn["LL"] entry (the per-connection LL cap removed)
    #   stripe_passes true: the engine's stripe loop (msccl_engine.cuh:285-426): a striped rank walks all of a TB's
    #               steps once per stripe, each pass moving span_k = floor(cnt (k+1) / nS) - floor(cnt k / nS)
    #               chunks of every step; one ET node per non-empty (pass, step); an empty call costs
    #               zero_stripe_ns before the TB's next call; a cross-TB dependency is waited for in pass 0 on the
    #               producer's LAST pass (flags are published only at the last stripe, cuh:295-317, 415-423); and
    #               every call takes DIVUP(span, maxAllowedCount) x chunkSteps FIFO slots (msccl_engine_host.cc:
    #               389-395, 490-503; maxAllowedCount = max(1, chunk_bytes / bytes per MSCCL chunk))
    # Stage 5c keys (calib/c20261105/stage5c/SPEC.md; each optional, absent = Stage 5b ETs byte for byte):
    #   nic_duplex  {"shared_GBps", "conn_factor": [[k, f], ...]}: every network send books its wire bytes on the sender's
    #               and the receiver's NIC duplex server (both directions of a NIC share shared_GBps; one direction alone
    #               is the YAML link rate): s5_nic_src_ns = W / (shared x f(k_tx(src))), s5_nic_dst_ns = W / (shared x
    #               f(k_rx(dst))); f = connection-count factor, k = the rank's network connections in that direction
    #               (piecewise linear in log2 k, flat outside the table)
    #   stripe_cap  int: nStripes = min(nStripes, cap) (runtime_v2c NCCL_MSCCL_MAX_STRIPES; validation of stripe A/Bs)
    #   fill_ns_per_B {proto: x} (sensitivity): + x * P ns of latency per network call (not pipelined across messages)
    #   pass_excess_ns (sensitivity): + this pre-delay on a TB's first call of every stripe pass k >= 1
    #   far_iter_ns {proto: x} (sensitivity): + x * min(1, P / 32 KiB) ns of latency per far network call
    #   ir_meta     "sha256": every rank's ET global metadata carries the IR's sha256 instead of its text (AstraSim does
    #               not read it); iters.max_nodes: adaptive unroll U = min(n_iter, max_unroll, max_nodes // calls of one
    #               iteration), at least 1
    #   iters       {"max_unroll": U}: the engine's outer loop (msccl_engine.cuh:258, every TB runs all of its steps
    #               for iteration i before iteration i + 1; chunk = 32 KiB LL / 2 MiB Simple per MSCCL chunk). The
    #               IR is unrolled min(n_iter, U) times in memory (TB step order across iterations, cross-TB deps
    #               within an iteration); when n_iter > U each copy stands for g = n_iter / U iterations and pays
    #               the per-message costs g times in series. nStripes come from the original IR; hand-off slots are
    #               the slots of one iteration

    @staticmethod
    def _s5_chassis_no(c: Any) -> int:
        return int(c) if isinstance(c, int) else int(str(c).lstrip("c"))

    def _s5_nodeclass(self, a: int, b: int) -> str:
        """Distance class of two nodes: S same chassis, C same cabinet / other chassis, X other cabinet;
        Stage 5b half-cabinet rule: N same cabinet and chassis half, F the rest."""
        if a == b:
            raise ValueError("same node")
        pl = self.stage5.get("placement") or {"mode": "uniform", "class": "S"}
        mode = pl["mode"]
        if mode == "uniform":
            return pl["class"]
        if mode == "blocks":
            npc, cpc = int(pl["nodes_per_chassis"]), int(pl["chassis_per_cabinet"])
            if a // npc == b // npc:
                return "S"
            return "C" if a // (npc * cpc) == b // (npc * cpc) else "X"
        if mode == "map":
            nodes = pl["nodes"]
            if max(a, b) >= len(nodes):
                raise ValueError(f"placement map has {len(nodes)} nodes, node {max(a, b)} requested")
            (ca, sa), (cb, sb) = nodes[a], nodes[b]
            if pl.get("rule") == "half_cabinet":
                near = ca == cb and self._s5_chassis_no(sa) // 4 == self._s5_chassis_no(sb) // 4
                return "N" if near else "F"
            if pl.get("rule") is not None:
                raise ValueError(f"placement rule {pl['rule']!r}")
            if ca == cb:
                return "S" if sa == sb else "C"
            return "X"
        raise ValueError(f"placement mode {mode!r}")

    @staticmethod
    def _s5_interp(table: list, x: float) -> float:
        pts = sorted(table)
        if x <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    def _s5_wire_factor(self, protocol: str) -> float:
        return self.ll_size_factor if protocol == "LL" else 1.0

    def _s5_regime(self, wire: float) -> str:
        n = self.stage5["net"]
        if wire <= n["idc_limit_B"]:
            return "idc"
        return "eager" if wire <= n["rdzv_threshold_B"] else "rdzv"

    def _s5_nmsg(self, protocol: str, payload: float) -> int:
        """Runtime network messages of one stripe call: one per FIFO slot (NCCL_STEPS buffer / 8) x messages per slot."""
        p = self.stage5.get("proxy")
        if not p:
            return 0
        slot = p["slot_B"][protocol]
        return max(1, -(-int(payload) // slot)) * p["msgs_per_slot"][protocol]

    def _s5_conn_ns(self, protocol: str, payload: float) -> float:
        c = (self.stage5.get("conn") or {}).get(protocol)
        if not c:
            return 0.0
        if "head_B" in c:
            first = min(payload, c["head_B"])
            return first / c["head_GBps"] + (payload - first) / c["GBps"]
        return payload / c["GBps"]

    def _s5_onpath(self, protocol: str, payload: float, wire: float) -> float:
        """Modelled pieces of a one-op network hop other than the fabric residual (ns), for one single-slot call:
        send work + proxy posting + route + max(wire / NIC, connection pace) + [Simple] proxy completion + flush
        + receive tail. Multi-slot work is pipelined (background) and does not enter the one-op hop."""
        s, n = self.stage5, self.stage5["net"]
        cost = n["send_w0_ns"][protocol] + n["tail_ns"][protocol] + 2 * self.inter_latency_ns
        rank = (s.get("ll_rank") or {}).get("GBps") if protocol == "LL" else None
        cost += max(wire / n["nic_GBps"], self._s5_conn_ns(protocol, payload), payload / rank if rank else 0.0)
        cost += (n.get("recv_flush_ns") or {}).get(protocol, 0)
        p = s.get("proxy")
        if p:
            k = p["msgs_per_slot"][protocol]
            cost += k * p["msg_ns"] * p["pre_frac"]                       # sender: posting work before injection
            if protocol == "Simple":                                       # receiver: the GPU waits for completion work
                cost += k * p["msg_ns"] * (1 - p["pre_frac"])
        return cost

    def _s5_proxy_split(self, protocol: str, P: float, payload: float, m: int, send: bool) -> tuple:
        """(pre, background, post) proxy work of one op: the first slot's posting, the other slots' work plus the
        per-byte work (pipelined with the stream), the last slot's completion."""
        p = self.stage5["proxy"]
        s = p["msgs_per_slot"][protocol]
        k = self._s5_nmsg(protocol, P)
        pre = m * s * p["msg_ns"] * p["pre_frac"]
        post = m * s * p["msg_ns"] * (1 - p["pre_frac"])
        bg = m * (k - s) * p["msg_ns"]
        if protocol == "LL" and send:
            bg += payload / p["ll_scan_GBps"]
        if protocol == "Simple":
            bg += payload / p["simple_byte_GBps"]
        return pre, bg, post

    def _s5_fabric_ns(self, protocol: str, payload: float, wire: float, dist: str) -> float:
        """Fabric / NIC / library latency of one runtime message at distance `dist` (the measured one-op hop minus the
        modelled pieces, interpolated in wire bytes within its regime, flat outside the measured points)."""
        n = self.stage5["net"]
        regime = self._s5_regime(wire)
        f = self._s5_wire_factor(protocol)
        pts = [[w, h - self._s5_onpath(protocol, w / f, w)] for w, h in n["hop_ns"][protocol][regime]]
        lat = self._s5_interp(pts, wire)
        if dist not in ("S", "N"):
            d = n["dist_ns"][protocol][dist]
            lat += d
            if regime == "rdzv":
                extra = (n.get("rdzv_dist_ns") or {}).get(protocol, {}).get(dist)
                lat += extra if extra is not None else (n["rdzv_traversals"] - 1) * d
        return lat

    def _s5_xtb(self, gpu_id: int, tb_id: int, step: Any, et_node: Any, kind: str, node_map: dict,
                protocol: str) -> None:
        x = self.stage5.get("xtb")
        if not x or kind not in ("s", "r", "rrc") or int(step.attrib['depid']) in (-1, tb_id):
            return
        dep = node_map[gpu_id][int(step.attrib['depid'])][int(step.attrib['deps'])]
        if type(dep) is MSCCLNopStep:
            return
        parent_id = dep.comp_node.id if type(dep) is MSCCLReceiveReduceComputeStep else dep.node.id
        chk, bind = x["chk_ns"][protocol], x["bind_ns"]
        if chk > bind:
            raise ValueError(f"xtb chk {chk} > bind {bind}: AstraSim's xtb_edge_delay needs chk <= bind")
        n = et_node.node if kind == "s" else self._recv_node(et_node)
        self._set_attr(n, "xtb_parent", parent_id)
        self._set_attr(n, "xtb_bind_ns", bind)
        self._set_attr(n, "xtb_chk_ns", chk)

    def apply_stage5(self, root: Any, protocol: str, node_map: dict, step_map: dict) -> None:
        s = self.stage5
        n, v = s["net"], s["nvl"]
        prx, hof, rrc = s.get("proxy"), s.get("handoff"), s.get("rrc")
        call = s.get("call")
        z_ns = (s.get("zero_stripe_ns") or {}).get(protocol, 0)
        f = self._s5_wire_factor(protocol)
        g = self._s5_iter_g                                    # Stage 5b iters: iterations per unrolled copy
        llr = s.get("ll_rank") if protocol == "LL" else None   # Stage 5b: per-rank LL data server
        nicd = s.get("nic_duplex")                              # Stage 5c: per-NIC duplex budget
        ktx, krx = self._s5_net_conns(root) if nicd else ({}, {})
        fill = (s.get("fill_ns_per_B") or {}).get(protocol, 0)
        far_it = (s.get("far_iter_ns") or {}).get(protocol, 0)
        pexc = s.get("pass_excess_ns", 0)
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            stripes = self._s5_stripes[gpu_id] if self._s5_stripes is not None else self._s5_nstripes(gpu, protocol)
            num_chunks = int(gpu.attrib['i_chunks']) or int(gpu.attrib['o_chunks'])
            chunk_size = int(self.collective_size / num_chunks)
            ll_conns = set()
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    step = step_map[gpu_id][tb_id][step_id]
                    kind = step.attrib['type']
                    if kind not in ("s", "r", "rrc"):
                        continue
                    cnt = int(step.attrib['cnt'])
                    m = max(1, msccl_engine_model.stripe_calls(cnt, stripes)) if stripes > 1 else 1
                    z = (stripes - m) * z_ns if stripes > 1 else 0
                    self._s5_xtb(gpu_id, tb_id, step, et_node, kind, node_map, protocol)
                    payload = chunk_size * cnt
                    # mc: calls this node stands for (stripe calls x iterations per unrolled copy), their per-message
                    # costs in series. Stage 5: mc = m. Stage 5b stripe_passes: m = 1, empty calls from _s5_unroll
                    mc = m * g if g != 1 else m
                    zc = self._s5_zcalls.get((gpu_id, tb_id, step_id)) if self._s5_zcalls else None
                    zpre, zpost = (zc[0] * z_ns * g, zc[1] * z_ns * g) if zc else (0, 0)
                    if pexc and (gpu_id, tb_id, step_id) in self._s5_passfirst:
                        zpre += pexc
                    if g != 1:
                        z = z * g
                    P = payload / mc
                    if kind == "s":
                        dst = int(tb.attrib['send'])
                        node = et_node.node
                        wire = self._attr(node, "comm_size").uint64_val
                        W = wire / mc
                        delay = z + zpre
                        if not self.is_network(gpu_id, dst):
                            # one TB copies at its pace; the link time W / GBps is added by the network
                            work = v["send_w0_ns"][protocol] + max(0.0, P / v["pace_GBps"][protocol] - W / v["GBps"])
                            resid = v["hop_ns"][protocol] - v["tail_ns"][protocol] - self.intra_latency_ns \
                                - 1024 * f / v["GBps"]
                            delay += mc * (work + max(0.0, resid)) + zpost
                            self._attr(node, "comm_delay_ns").uint64_val = int(round(delay))
                            continue
                        self._attr(node, "comm_delay_ns").uint64_val = int(round(delay))
                        self._set_attr(node, "s5_net", 1)
                        dist = self._s5_nodeclass(gpu_id // self.gpus_per_node, dst // self.gpus_per_node)
                        work0 = n["send_w0_ns"][protocol]
                        self._set_attr(node, "s5_work_ns", round(mc * (work0 + P / n["send_GBps"][protocol]) + zpost))
                        self._set_attr(node, "s5_post_after_ns", round(mc * work0))
                        conn = self._s5_conn_ns(protocol, payload)
                        Wm = min(W, self._s5_slot_wire(protocol))
                        lat = mc * self._s5_fabric_ns(protocol, P, Wm, dist)
                        if fill:
                            lat += mc * fill * P
                        if far_it and dist not in ("S", "N"):
                            lat += mc * far_it * min(1.0, P / 32768)
                        if nicd:
                            sh = nicd["shared_GBps"]
                            self._set_attr(node, "s5_nic_src_ns", round(wire / (sh * self._s5_conn_factor(
                                nicd["conn_factor"], ktx.get(gpu_id, 1)))))
                            self._set_attr(node, "s5_nic_dst_ns", round(wire / (sh * self._s5_conn_factor(
                                nicd["conn_factor"], krx.get(dst, 1)))))
                        self._set_attr(node, "s5_conn_ns", round(conn))
                        self._set_attr(node, "s5_inject_ns", max(0, round(lat)))
                        self._set_attr(node, "s5_route_ns", round(2 * self.inter_latency_ns))
                        if llr:
                            rank_ns = payload / llr["GBps"]
                            ckey = (dst, et_node.tag)
                            if ckey not in ll_conns:
                                if ll_conns and llr.get("extra_conn_ns"):
                                    rank_ns += llr["extra_conn_ns"]
                                ll_conns.add(ckey)
                            self._set_attr(node, "s5_rank_ns", round(rank_ns))
                        occ = (n.get("rdzv_occ_ns") or {}).get(protocol)
                        if occ and self._s5_regime(Wm) == "rdzv":
                            self._set_attr(node, "s5_conn_occ_ns", round(mc * occ))
                        if prx:
                            pre, bg, post = self._s5_proxy_split(protocol, P, payload, mc, True)
                            self._set_attr(node, "s5_px_pre_ns", round(pre))
                            self._set_attr(node, "s5_px_bg_ns", round(bg))
                            self._set_attr(node, "s5_px_post_ns", round(post))
                        if hof:
                            n_iter = 1 if s.get("iters") else max(1, -(-chunk_size // hof["chunk_bytes"][protocol]))
                            if s.get("stripe_passes"):   # per call: DIVUP(span, maxAllowedCount) x chunkSteps
                                allowed = max(1, hof["chunk_bytes"][protocol] // max(1, self._s5_chunk_B))
                                slots = -(-cnt // allowed) * hof["chunk_steps"][protocol] * n_iter
                            else:
                                slots = -(-cnt // stripes) * hof["chunk_steps"][protocol] * n_iter
                            self._set_attr(node, "handoff_depth", hof["depth"])
                            self._set_attr(node, "handoff_slots", slots if slots <= hof["depth"] else 0)
                    else:
                        src = int(tb.attrib['recv'])
                        rn = self._recv_node(et_node)
                        net = self.is_network(src, gpu_id)
                        tail = (n if net else v)["tail_ns"][protocol]
                        self._set_attr(rn, "comm_post_delay_ns", round(mc * tail + z + zpost))
                        if zpre:
                            d = self._attr(rn, "comm_delay_ns")
                            d.uint64_val = int(d.uint64_val + round(zpre))
                        if net:
                            self._set_attr(rn, "s5_net", 1)
                            flush = (n.get("recv_flush_ns") or {}).get(protocol, 0)
                            if flush:
                                self._set_attr(rn, "s5_rx_lat_ns", round(mc * flush))
                            if prx:
                                pre, bg, post = self._s5_proxy_split(protocol, P, payload, mc, False)
                                self._set_attr(rn, "s5_px_pre_ns", round(pre))
                                self._set_attr(rn, "s5_px_bg_ns", round(bg))
                                self._set_attr(rn, "s5_px_post_ns", round(post))
                            self._set_attr(rn, "s5_px_wait", 1 if protocol == "Simple" else 0)
                        if kind == "rrc":
                            r = (rrc or {}).get(protocol, {}).get("net" if net else "nvl") if rrc is not None else "legacy"
                            if r == "legacy":
                                continue
                            ns = 0.0 if r is None else \
                                mc * min(P, r["slot_B"]) * (1 / r["rrc_GBps"] - 1 / r["copy_GBps"])
                            et_node.comp_node.duration_micros = 0
                            et_node.comp_ns = ns
            # per call: every root step of the rank waits for the call cost (kernel gap + TB setup)
            if call:
                c_ns = call["gap_ns"] + call["setup_ns"][protocol]
                for steps in node_map[gpu_id].values():
                    for et_node in steps.values():
                        if type(et_node) is MSCCLNopStep:
                            continue
                        node = self._recv_node(et_node) if type(et_node) is not MSCCLSendStep else et_node.node
                        if len(node.data_deps) == 0:
                            d = self._attr(node, "comm_delay_ns")
                            d.uint64_val = int(d.uint64_val + c_ns)

    def _s5_nstripes(self, gpu: Any, protocol: str) -> int:
        n = self._n_stripes(gpu, protocol)
        cap = self.stage5.get("stripe_cap") if self.stage5 is not None else None
        return min(n, int(cap)) if cap else n

    @staticmethod
    def _s5_conn_factor(table: list, k: int) -> float:
        import math
        pts = sorted([math.log2(max(1, a)), b] for a, b in table)
        x = math.log2(max(1, k))
        if x <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    def _s5_net_conns(self, root: Any) -> tuple:
        """{rank: distinct network send connections (dst, tag)}, {rank: distinct network receive connections (src, tag)}."""
        slots = {}
        tx, rx = {}, {}
        for gpu in root.findall('gpu'):
            g = int(gpu.attrib['id'])
            cidx = msccl_engine_model.conn_index(gpu) if self.msccl_conn_tags else {}
            for tb in gpu.findall('tb'):
                t = int(tb.attrib['id'])
                chan = int(tb.attrib['chan'])
                rs, ss = cidx.get(t, (0, 0)) if cidx else (0, 0)
                types = {st.attrib['type'] for st in tb.findall('step')}
                snd, rcv = int(tb.attrib['send']), int(tb.attrib['recv'])
                if snd >= 0 and types & msccl_engine_model.SEND_TYPES and self.is_network(g, snd):
                    tx.setdefault(g, set()).add((snd, msccl_engine_model.conn_tag(chan, ss) if ss else chan))
                if rcv >= 0 and types & msccl_engine_model.RECV_TYPES and self.is_network(rcv, g):
                    rx.setdefault(g, set()).add((rcv, msccl_engine_model.conn_tag(chan, rs) if rs else chan))
        return {g: len(v) for g, v in tx.items()}, {g: len(v) for g, v in rx.items()}

    def _s5_unroll(self, root: Any, protocol: str) -> None:
        """Stage 5b: rewrite the parsed IR into the engine's call order (msccl_engine.cuh:258-426):
            for iteration j (outer loop, chunk_bytes per MSCCL chunk) / for stripe pass k / for each step of the TB.
        iters: U = min(n_iter, max_unroll) copies; each moves 1 / U of the bytes (every rank's chunk count x U) and
        stands for g = n_iter / U iterations. stripe_passes: per rank with nStripes = nS > 1 (from the original IR, as
        the engine computes it once per call), pass k of a step of cnt chunks keeps span_k = floor(cnt (k+1) / nS) -
        floor(cnt k / nS) chunks; an empty call (span 0) becomes no node and is counted for the TB's next call
        (zero_stripe_ns), or for its last call when it is trailing. A step's cross-TB dependency is waited for only in
        pass 0, on the producer's copy in the LAST pass (or the producer's last non-empty call before it); copies in
        passes > 0 have none (flags are published at the last stripe, cuh:295-317, 415-423). Without stripe_passes a
        copy keeps the dependency of its own iteration (flag = (iter, step), cuh:49-51).
        msccl_stripe_runtime "v3" (runtime_v3 3228413, msccl_engine_model.py part 4): on a rank that uses per-stripe
        flags, a step with an aligned dependency waits in EVERY pass k, on the producer's copy in pass k (or the
        producer's last non-empty call before it); its other dependencies and all other ranks keep the rule above
        (device/msccl_engine.cuh:305-332 wait, :434 publish; msccl_engine_host.cc:886-895 flag choice)."""
        s5 = self.stage5
        it = s5.get("iters")
        passes = bool(s5.get("stripe_passes"))
        gpus = root.findall('gpu')
        orig = {int(gp.attrib['id']): self._s5_nstripes(gp, protocol) for gp in gpus}
        ncpl = int(root.attrib.get('nchunksperloop') or 0) or \
            max(int(gp.attrib['i_chunks']) or int(gp.attrib['o_chunks']) for gp in gpus)
        self._s5_chunk_B = int(self.collective_size // ncpl)
        U = 1
        if it and 'allgather' not in self.input_filename:   # the allgather chunk-size convention: no unroll
            n_iter = max(1, -(-self._s5_chunk_B // int(it["chunk_bytes"][protocol])))
            U = max(1, min(n_iter, int(it["max_unroll"])))
            if it.get("max_nodes"):
                # Stage 5c: adaptive unroll, so that the ET has at most max_nodes call nodes (all ranks): base = the
                # calls of one iteration (after stripe passes)
                base = 0
                for gp in gpus:
                    nS_ = orig[int(gp.attrib['id'])] if passes else 1
                    for tb in gp.findall('tb'):
                        for st in tb.findall('step'):
                            if st.attrib['type'] in ('s', 'r', 'rrc'):
                                cnt = int(st.attrib['cnt'])
                                base += sum(1 for k in range(nS_) if (cnt * (k + 1)) // nS_ - (cnt * k) // nS_ > 0)
                U = max(1, min(U, int(it["max_nodes"]) // max(1, base)))
            self._s5_iter_g = n_iter / U if n_iter != U else 1
        self._s5_stripes = {gid: (1 if passes else nS) for gid, nS in orig.items()}
        self._s5_zcalls = {}
        v3_flags = passes and self._launch_stripes is not None   # runtime_v3 NCCL_MSCCL_STRIPE_FLAGS=auto
        flag_cache = {}
        if U == 1 and not (passes and any(nS > 1 for nS in orig.values())):
            return
        if U > 1 and root.attrib.get('nchunksperloop'):
            root.set('nchunksperloop', str(int(root.attrib['nchunksperloop']) * U))
        for gp in gpus:
            gid = int(gp.attrib['id'])
            nS = orig[gid] if passes else 1
            if U > 1:
                for key in ('i_chunks', 'o_chunks'):
                    if int(gp.attrib[key]):
                        gp.set(key, str(int(gp.attrib[key]) * U))
            if U == 1 and nS == 1:
                continue
            tbs = gp.findall('tb')
            steps = {int(tb.attrib['id']): sorted(tb.findall('step'), key=lambda st: int(st.attrib['s'])) for tb in tbs}
            # runtime_v3: (tb id, step index) of the steps whose dependency waits in every pass (aligned marks of
            # mscclStripeAnalyze; empty unless this rank's v2b flag order has a cycle that the marks remove)
            aligned = frozenset()
            if v3_flags and nS > 1:
                rtbs = msccl_engine_model.rank_tbs(gp)
                sig = (msccl_engine_model.rank_sig(rtbs), nS)
                if sig not in flag_cache:
                    flag_cache[sig] = msccl_engine_model.stripe_flags(rtbs, nS, "auto")
                tb_ids = sorted(steps)                 # rank_tbs() index -> tb id
                aligned = frozenset((tb_ids[b], i) for b, i in flag_cache[sig][1])
                self._v3_info["stripe_flag_ranks"] += 1 if flag_cache[sig][0] else 0
                self._v3_info["aligned_deps"] += len(aligned)
            # new index of every kept (j, k, i) copy, and the TB's last non-empty call at or before it (-1: none).
            # An empty pass-0 copy of a step with a dependency is kept as a nop that carries the dependency: the
            # engine waits for it in pass 0 whatever the span (cuh:300-317 run before the span test, cuh:335).
            newidx, at_or_before, kind = {}, {}, {}
            for tb_id, sts in steps.items():
                nxt, last = 0, -1
                for j in range(U):
                    for k in range(nS):
                        for i, st in enumerate(sts):
                            cnt = int(st.attrib['cnt'])
                            span = (cnt * (k + 1)) // nS - (cnt * k) // nS
                            waits = k == 0 or (tb_id, i) in aligned   # this pass waits for the step's dependency
                            if st.attrib['type'] == 'nop':
                                kd = 'nop' if waits else None
                            elif span > 0:
                                kd = 'real'
                            else:
                                kd = 'waitnop' if waits and int(st.attrib['depid']) != -1 else None
                            if kd:
                                newidx[(tb_id, j, k, i)] = nxt
                                kind[(tb_id, j, k, i)] = kd
                                if kd == 'real':
                                    last = nxt
                                nxt += 1
                            at_or_before[(tb_id, j, k, i)] = last
            for tb in tbs:
                tb_id = int(tb.attrib['id'])
                sts = steps[tb_id]
                for st in sts:
                    tb.remove(st)
                pend, last_new = 0, None
                for j in range(U):
                    for k in range(nS):
                        first_in_pass = k > 0
                        for i, st in enumerate(sts):
                            key = (tb_id, j, k, i)
                            kd = kind.get(key)
                            if kd != 'real' and st.attrib['type'] != 'nop':
                                pend += 1        # an empty stripe call of a real step
                            if kd is None:
                                continue
                            c = ElementTree.Element(st.tag, dict(st.attrib))
                            c.tail = st.tail
                            c.set('s', str(newidx[key]))
                            if kd == 'waitnop':
                                c.set('type', 'nop')
                                c.set('cnt', '0')
                            elif nS > 1 and kd == 'real':
                                cnt = int(st.attrib['cnt'])
                                c.set('cnt', str((cnt * (k + 1)) // nS - (cnt * k) // nS))
                            dep = int(st.attrib['depid'])
                            if dep != -1:
                                if (tb_id, i) in aligned:      # producer's same pass (cuh:319-320, goal subIter)
                                    tgt = at_or_before[(dep, j, k, int(st.attrib['deps']))]
                                else:                          # pass 0 only, producer's last pass (lastSubIter)
                                    tgt = at_or_before[(dep, j, nS - 1, int(st.attrib['deps']))] if k == 0 else -1
                                if tgt < 0:
                                    c.set('depid', '-1')
                                    c.set('deps', '-1')
                                else:
                                    c.set('deps', str(tgt))
                            tb.append(c)
                            if kd != 'real':
                                continue
                            last_new = newidx[key]
                            if first_in_pass:
                                self._s5_passfirst.add((gid, tb_id, last_new))
                                first_in_pass = False
                            if pend:
                                self._s5_zcalls[(gid, tb_id, last_new)] = [pend, 0]
                                pend = 0
                if pend and last_new is not None:
                    self._s5_zcalls.setdefault((gid, tb_id, last_new), [0, 0])[1] += pend

    def _s5_slot_wire(self, protocol: str) -> float:
        """Wire bytes of one runtime network message (one FIFO slot); the message's latency regime uses it."""
        p = self.stage5.get("proxy")
        slot = (p or {}).get("slot_B", {}).get(protocol, 1 << 62)
        return slot * self._s5_wire_factor(protocol)

    def s5_copies(self, nodes: list) -> list:
        """Back-to-back calls (stage5 'calls' = K > 1): K copies of one rank's ET nodes. Copy j has ids + j * N; its
        root nodes (no data parent) also depend on every leaf of copy j - 1 (kernel order on one stream)."""
        K = int(self.stage5.get("calls", 1))
        if K <= 1 or not nodes:
            return nodes
        N = max(nd.id for nd in nodes) + 1
        parents = {d for nd in nodes for d in nd.data_deps}
        leaves = [nd.id for nd in nodes if nd.id not in parents]
        out = list(nodes)
        for j in range(1, K):
            for nd in nodes:
                c = Node()
                c.CopyFrom(nd)
                c.id = nd.id + j * N
                deps = [d + j * N for d in nd.data_deps]
                if not deps:
                    deps = [l + (j - 1) * N for l in leaves]
                del c.data_deps[:]
                c.data_deps.extend(deps)
                for a in c.attr:
                    if a.name == "xtb_parent":
                        a.uint64_val = a.uint64_val + j * N
                out.append(c)
        return out

    # Creates the global metadata info that is added to the start of all ET files.
    def create_global_metadata(self):
        input_text = ""
        if self.stage5 is not None and self.stage5.get("ir_meta") == "sha256":
            # Stage 5c: the IR's sha256 instead of its text in every rank's ET (AstraSim does not read input_file;
            # at 256 GPUs the text is > 99 % of the ET bytes and of the sim's memory)
            if getattr(self, "_s5_ir_sha", None) is None:
                import hashlib
                with open(self.input_filename, "rb") as input_file:
                    self._s5_ir_sha = hashlib.sha256(input_file.read()).hexdigest()
            input_text = "sha256:" + self._s5_ir_sha
        else:
            with open(self.input_filename, "r") as input_file:
                input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text)
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata

    # Creates an ET node, and assigns a node id to it.
    # Increment the node id, to be assigned to the next ET node.
    def get_et_node_id(self) -> int:
        id = self.next_node_id
        self.next_node_id += 1
        return id

    # This function is called to reset the node id when starting to add nodes for a new ET trace file.
    # There will be one ET trace file for each NPU. 
    def reset_node_id(
            self
    ): self.next_node_id = 0

    # Add 'parent_node' as the parent to 'child_node'.
    # Note that parent_node and child_node has to be within the same ET trace file.
    def add_parent(
        self,
        child_node: Any,
        parent_node: Any
    ) -> None:
        child_node.data_deps.append(parent_node.id)
        
    def convert(self) -> None:
        node_map = {}
        step_map = {}
        tree = ElementTree.parse(self.input_filename)
        root = tree.getroot()
        protocol = self.protocol_override or root.attrib.get("proto", "Simple")
        self._set_launch_stripes(root, protocol)
        if self.stage5 is not None and (self.stage5.get("iters") or self.stage5.get("stripe_passes")):
            self._s5_unroll(root, protocol)

        # Read the XML file and create ET Trace nodes. 
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            num_chunks = int(gpu.attrib['i_chunks'])
            if num_chunks == 0:
                num_chunks = int(gpu.attrib['o_chunks'])
            chunk_size = int(self.collective_size / num_chunks)
            ## HARDCODED. REMOVE WHEN IMPLEMENT COLLECTIVE API
            if 'allgather' in self.input_filename:
                # collective_size is each rank's contribution (the convention for
                # one chunk per rank). In-place AllGather IRs have i_chunks=0 and
                # o_chunks = ngpus * chunks_per_rank, where chunks_per_rank grows with
                # `instances` / chunk_factor. The old hardcode gave EVERY chunk the
                # full collective_size, so k chunks per rank sent k times the bytes.
                ngpus = int(root.attrib['ngpus'])
                chunks_per_rank = max(1, int(gpu.attrib['o_chunks']) // ngpus)
                chunk_size = int(self.collective_size / chunks_per_rank)
            node_map[gpu_id] = {}
            step_map[gpu_id] = {}
            self.reset_node_id()
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                node_map[gpu_id][tb_id] = {}
                step_map[gpu_id][tb_id] = {}
                for step in tb.findall('step'):
                    step_id = int(step.attrib['s'])
                    chunk_cnt = int(step.attrib['cnt'])
                    payload_size = chunk_size * chunk_cnt
                    if step.attrib['type'] == "s":
                        comm_size = self.effective_comm_size(protocol, payload_size, gpu_id, int(tb.attrib['send']))
                    elif step.attrib['type'] in ("r", "rrc"):
                        comm_size = self.effective_comm_size(protocol, payload_size, int(tb.attrib['recv']), gpu_id)
                    else:
                        comm_size = self.effective_comm_size(protocol, payload_size)
                    step_map[gpu_id][tb_id][step_id] = step
                    et_node_id = self.get_et_node_id()
                    if step.attrib['type'] == "s":
                        dst_rank = int(tb.attrib['send'])
                        comm_delay_ns = self.protocol_send_delay_ns(protocol, gpu_id, dst_rank)
                        node = MSCCLSendStep(
                            tb, step_id, et_node_id, comm_size, protocol, comm_delay_ns)
                        node.payload_bytes = payload_size
                        node_map[gpu_id][tb_id][step_id] = node
                    elif step.attrib['type'] == "r":
                        node = MSCCLReceiveStep(tb, step_id, et_node_id, comm_size, protocol)
                        node_map[gpu_id][tb_id][step_id] = node
                    elif step.attrib['type'] == "rrc":
                        comp_et_node_id = self.get_et_node_id()
                        node = MSCCLReceiveReduceComputeStep(
                            tb, step_id, et_node_id, comp_et_node_id,
                            comm_size, payload_size, protocol)
                        node.payload_bytes = payload_size
                        node_map[gpu_id][tb_id][step_id] = node
                    elif step.attrib['type'] == "nop":
                        node = MSCCLNopStep()
                        node_map[gpu_id][tb_id][step_id] = node

        # For each ET Trace node, add the parent dependency information
        for gpu_id in node_map:
            for tb_id in node_map[gpu_id]:
                # Note that we're doing this in reverse order within a threadblock.
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    if type(et_node) is MSCCLNopStep:
                        continue
                    step = step_map[gpu_id][tb_id][step_id]
                    dep_tb_id = int(step.attrib['depid'])
                    dep_step_id = int(step.attrib['deps'])

                    # Parent by data dependency
                    if dep_tb_id != -1:
                        dep_node = node_map[gpu_id][dep_tb_id][dep_step_id]
                        try:
                            et_node.add_parent(dep_node)
                        except Exception as e:
                            print(e, f'{gpu_id}, {dep_tb_id}, {dep_tb_id}, {tb_id}, {step_id}')

                    # Parent by control
                    if step_id != 0:
                        prev_step_id = step_id -1
                        #print(gpu_id, tb_id, prev_step_id, step_id)
                        prev_node = node_map[gpu_id][tb_id][prev_step_id]
                        while type(prev_node) is MSCCLNopStep:
                            prev_step = step_map[gpu_id][tb_id][prev_step_id]
                            dep_tb_id = int(prev_step.attrib['depid'])
                            dep_step_id = int(prev_step.attrib['deps'])
                            if dep_tb_id != -1:   # -1: a wait whose producer has no call yet (stripe passes)
                                dep_node = node_map[gpu_id][dep_tb_id][dep_step_id]
                                et_node.add_parent(dep_node)

                            prev_step_id = prev_step_id - 1
                            if prev_step_id < 0:
                                break
                            prev_node = node_map[gpu_id][tb_id][prev_step_id]

                        if type(prev_node) is not MSCCLNopStep:
                            et_node.add_parent(prev_node)

        if self.msccl_conn_tags:
            self.apply_conn_tags(root, node_map, step_map)
        if self.stage5 is not None:
            self.apply_stage5(root, protocol, node_map, step_map)
            self.apply_comp_ns(node_map)
            self.write_ets(node_map)
            return
        if self.stage4 is not None:
            self.apply_stage4_pre(root, protocol, node_map, step_map)
        if self.proxy_chan_ns is not None:
            self.apply_proxy_chan(root, protocol, node_map, step_map)
        if self.msccl_stripe_calls:
            self.apply_stripe_calls(root, protocol, node_map, step_map)
        if self.ll_conn_GBps is not None:
            self.apply_ll_conn_rate(root, protocol, node_map, step_map)
        if self.call_cost_ns is not None:
            self.apply_call_cost(protocol, node_map)
        if self.stage4 is not None:
            self.apply_stage4_post(root, protocol, node_map, step_map)
            self.apply_stage4b_post(root, protocol, node_map, step_map)
        if self.comp_ns:
            self.apply_comp_ns(node_map)
        if self.net_send_handoff is not None:
            self.apply_net_send_handoff(root, protocol, node_map, step_map)

        # For each ET Trace node, add the parent dependency information
        for gpu_id in node_map:
            output_filename = "%s.%s.et" % (self.output_filename, gpu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.create_global_metadata()
                encode_message(g, global_metadata)
                for tb_id in node_map[gpu_id]:
                    for step_id, et_node in node_map[gpu_id][tb_id].items():
                        if type(et_node) is MSCCLNopStep :
                            continue

                        et_node.encode_message(g)
                        #if gpu_id == 0:
                            #print('encode node', et_node)

    def write_ets(self, node_map: dict) -> None:
        """Stage 5 writer: the same node order as convert(), then the back-to-back copies (s5_copies)."""
        for gpu_id in node_map:
            nodes = []
            for tb_id in node_map[gpu_id]:
                for step_id, et_node in node_map[gpu_id][tb_id].items():
                    if type(et_node) is MSCCLNopStep:
                        continue
                    if type(et_node) is MSCCLReceiveReduceComputeStep:
                        nodes += [et_node.recv_node, et_node.comp_node]
                    else:
                        nodes.append(et_node.node)
            if not nodes:
                # harness padding: a rank whose TBs are all nop has no ET node, and AstraSim aborts on an empty ET
                # ("No dependancy free nodes found"). One zero-duration COMP node; the rank exchanges nothing.
                idle = Node()
                idle.id, idle.name, idle.type = 0, "IDLE_RANK_NOP", COMP_NODE
                nodes = [idle]
            with open("%s.%s.et" % (self.output_filename, gpu_id), "wb") as g:
                encode_message(g, self.create_global_metadata())
                for node in self.s5_copies(nodes):
                    encode_message(g, node)
