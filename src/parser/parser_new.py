import re
import numpy as np
from typing import Optional
from llvmlite.binding import parse_assembly, ValueRef

from .utils import find_all_values, tokenize_ir, find_all_identifiers
from .enums import opcode_to_meta


class IRUnit(object):
    # def __init__(self, string: str):
    #     self.string: str = string
    #     # all identifiers, note that it includes basic block and duplicated names
    #     self.identifiers = find_all_identifiers(self.string)

    def is_bb_name(self, identifier: str):
        assert hasattr(self, "bb_names")
        s = identifier.strip(" %@")
        if s in getattr(self, "bb_names"):
            return True
        else:
            return False


class Instruction(IRUnit):
    def __init__(self, instruction: ValueRef, bb_names: list[str], is_terminal: bool = False):
        super().__init__()
        self.string = " ".join(str(instruction).split())
        # self.tokens = tokenize_ir(self.string)
        self.identifiers = find_all_identifiers(self.string)

        # if this instruction is a return instruction
        self.is_return = False
        # ret, resume
        self.ret_type = None
        self.ret_value = None
        # if this instruction is a terminator
        self.is_terminator = is_terminal
        # the next jumps, used for CFG, {bb_name: cfg_type}
        self.jumps = {}
        # dfg edges, list of (from, to, dfg_type)
        self.data_flows = []
        # call meta data, (return_value, callee, args_list, [normal label, exception label])
        self.call = None
        # all basic block names, no % prefix
        self.bb_names = bb_names
        # parse
        self.parse()

        # print(self.string)
        # if self.is_terminator:
        #     print(self.jumps)
        # print("=" * 20)

    def add_jump(self, jump_to: str, ty: str):
        # identifiers of all bb labels should not include the % and @ prefixes.
        self.jumps.update({jump_to.strip("%@ "): ty})

    def update_assignment_data_flows(
        self,
        opcode: str,
        num_operands: int,
        non_com_opcode: bool,
        operation_postfixes: list[str] = None,
    ):
        if "=" not in self.string:
            return
        left, right = self.string.split("=")[:2]
        result = find_all_identifiers(left)[0]
        operands = find_all_values(right)
        if 0 < num_operands < len(operands):
            operands = operands[:num_operands]
        for idx, operand in enumerate(operands):
            if not self.is_bb_name(operand):
                if not non_com_opcode:
                    ty = opcode
                elif operation_postfixes:
                    postfix = (
                        operation_postfixes[idx]
                        if len(operation_postfixes) > idx
                        else operation_postfixes[-1]
                    )
                    ty = f"{opcode}.{postfix}"
                else:
                    ty = f"{opcode}.op{idx + 1}"

                self.data_flows.append((operand, result, ty))

    def parse(self):
        tokens = tokenize_ir(self.string)

        # parse CFG
        if self.is_terminator and len(tokens) > 2:
            if tokens[0] == "ret" or tokens[0] == "resume":
                self.is_return = True
                self.ret_type = tokens[0]
                # ids = find_all_identifiers(self.string)
                if len(self.identifiers) > 0:
                    self.ret_value = self.identifiers[0]
            elif tokens[0] == "br":
                if tokens[1] == "label":
                    self.add_jump(tokens[2], "br")
                else:
                    match = re.match(
                        r"br .+?, label ([%@][-a-zA-Z$._0-9]+), label ([%@][-a-zA-Z$._0-9]+)",
                        self.string,
                    )
                    self.add_jump(match.group(1), "br.T")
                    self.add_jump(match.group(2), "br.F")
            elif tokens[0] == "switch":
                match = re.match(
                    r"switch i\w+ .+?, label ([%@][-a-zA-Z$._0-9]+) \[(.+?)]",
                    self.string,
                    flags=re.S,
                )
                self.add_jump(match.group(1), "switch.default")
                case_jumps = match.group(2).strip()
                if case_jumps != "":
                    matches = re.findall(r"[%@][-a-zA-Z$._0-9]+", case_jumps)
                    for m in matches:
                        self.add_jump(m, "switch.case")
            elif tokens[0] == "indirectbr":
                matches = re.findall(r"label ([%@][-a-zA-Z$._0-9]+)", self.string)
                for m in matches:
                    self.add_jump(m, "indirectbr")
            elif tokens[2] == "catchswitch":
                match = re.match(
                    r"[%@][-a-zA-Z$._0-9]+ = catchswitch within .+? \[(.+?)] unwind (.+)",
                    self.string,
                )
                handlers = re.findall(r"label ([%@][-a-zA-Z$._0-9]+)", match.group(1))
                for handler in handlers:
                    self.add_jump(handler, "catchswitch.handler")
                if "label" in match.group(2):
                    default = re.search(r"[%@][-a-zA-Z$._0-9]+", match.group(2))
                    if default:
                        self.add_jump(default.group(), "catchswitch.default")
            elif tokens[0] == "catchret":
                match = re.search(r"label ([%@][-a-zA-Z$._0-9]+)", self.string)
                if match:
                    self.add_jump(match.group(), "catchret")
            elif tokens[0] == "cleanupret":
                match = re.search(r"label ([%@][-a-zA-Z$._0-9]+)", self.string)
                if match:
                    self.add_jump(match.group(), "cleanupret")
        # parse data flow
        if len(self.identifiers) > 1:
            data_flow_parsed = False
            for opcode, meta in opcode_to_meta.items():
                if opcode in tokens:
                    data_flow_parsed = True
                    self.update_assignment_data_flows(
                        opcode=meta[0],
                        num_operands=meta[1],
                        non_com_opcode=meta[2],
                        operation_postfixes=meta[3] if len(meta) > 3 else None,
                    )
            if not data_flow_parsed:
                if "store" in tokens:
                    ids = self.identifiers[:2]
                    self.data_flows.append((ids[0], ids[1], "store"))
                elif "icmp" in tokens or "fcmp" in tokens:
                    opcode = "icmp" if "icmp" in tokens else "fcmp"
                    left, right = self.string.split("=")[:2]
                    result = find_all_identifiers(left)[0]
                    operands = find_all_values(right)
                    if len(operands) > 2:
                        operands = operands[:2]
                    # find condition
                    cmp_idx = tokens.index(opcode)
                    condition = tokens[cmp_idx + 1]
                    # skip fast-math flags for fcmp
                    while opcode == "fcmp" and condition in [
                        "nnan",
                        "ninf",
                        "nsz",
                        "arcp",
                        "contract",
                        "afn",
                        "reassoc",
                        "fast",
                    ]:
                        cmp_idx += 1
                        condition = tokens[cmp_idx + 1]
                    for idx, operand in enumerate(operands):
                        if not self.is_bb_name(operand):
                            if condition in ["true", "false", "ord", "uno"]:
                                ty = f"cmp.{condition}"
                            else:
                                condition = condition[-2:]
                                if condition in ["eq", "ne"]:
                                    ty = f"cmp.{condition}"
                                else:
                                    ty = f"cmp.{condition}.op{idx + 1}"
                            self.data_flows.append((operand, result, ty))
            # parse external call
            if "call" in tokens:
                match = re.search(
                    r"([%@][-a-zA-Z$._0-9]+)?\s*=?\s*(tail|musttail|notail)?\s*call.+?(@[-a-zA-Z$._0-9]+)\((.*?)\)",
                    self.string,
                )
                if match is not None:
                    result = match.group(1)
                    callee = match.group(3)
                    arg_list = find_all_values(match.group(4))
                    if callee is not None:
                        self.call = (result, callee, arg_list)
            elif "invoke" in tokens:
                match = re.search(
                    r"([%@][-a-zA-Z$._0-9]+)?\s*=?\s*invoke.+?(@[-a-zA-Z$._0-9]+)\((.*?)\) to label ([%@]["
                    r"-a-zA-Z$._0-9]+) unwind label ([%@][-a-zA-Z$._0-9]+)",
                    self.string,
                )
                if match is not None:
                    result = match.group(1)
                    callee = match.group(2)
                    arg_list = find_all_values(match.group(3))
                    normal_label = match.group(4)
                    exception_label = match.group(5)
                    self.call = (
                        result,
                        callee,
                        arg_list,
                        normal_label,
                        exception_label,
                    )


class BasicBlock(IRUnit):
    def __init__(self, block: ValueRef, bb_names: list[str]):
        super().__init__()
        string = str(block).strip(" \n\r\t{}")
        self.name = block.name
        self.identifiers = find_all_identifiers(string)

        instructions = [instruction for instruction in block.instructions]
        self.instructions = []
        for i, instruction in enumerate(instructions):
            inst = Instruction(
                instruction,
                bb_names=bb_names,
                is_terminal=True if i == len(instructions) - 1 else False,
            )
            self.instructions.append(inst)


class Function(IRUnit):
    def __init__(self, function: ValueRef):
        super().__init__()
        string = str(function).strip()
        self.name = function.name

        self.args_string = ', '.join([str(arg) for arg in function.arguments])
        self.args = find_all_identifiers(self.args_string)

        self.bb_names = [block.name for block in function.blocks]

        self.basic_blocks = []
        for block in function.blocks:
            bb = BasicBlock(block, self.bb_names)
            self.basic_blocks.append(bb)

        self.variables = []
        for variable in self.args + find_all_identifiers(string):
            if variable not in self.variables and not self.is_bb_name(variable):
                self.variables.append(variable)

    def add_fun_prefix(self, base_str: str):
        if base_str.startswith("@"):
            return base_str
        else:
            return f"{self.name}-{base_str}"


class IRFile(IRUnit):
    def __init__(self, ir_str: str, cfg_vocab, dfg_vocab, build_graph=True):
        super().__init__()
        self.cfg_vocab = cfg_vocab
        self.dfg_vocab = dfg_vocab

        ir = parse_assembly(ir_str)

        self.functions = []
        for function in ir.functions:
            if function.is_declaration:
                continue
            func = Function(function)
            self.functions.append(func)

        self.graph_built = False
        if build_graph:
            # cfg, dfg, and bb-var matrices
            # do use `.get(name, default=-1)` to ensure the default value is -1, means no edge.
            # do use `if idx is not None` in case that the idx is 0, which is valid edge.
            self.bb_to_idx = {}
            bb_idx = 0
            self.id_to_idx = {}
            id_idx = 0
            self.var_list = []
            for func in self.functions:
                # cfg
                # function signature is a special BB
                self.bb_to_idx.update({func.name: bb_idx})
                bb_idx += 1
                # BBs
                for bb in func.basic_blocks:
                    self.bb_to_idx.update({func.add_fun_prefix(bb.name): bb_idx})
                    bb_idx += 1
                # dfg
                self.id_to_idx.update({func.name: id_idx})
                self.var_list.append(func.name)
                id_idx += 1
                # arguments
                for var in func.variables:
                    var_name = func.add_fun_prefix(var)
                    if var_name not in self.id_to_idx:
                        self.id_to_idx.update({var_name: id_idx})
                        id_idx += 1
                        self.var_list.append(var)
                # for arg in func.args:
                #     self.id_to_idx.update({func.add_fun_prefix(arg): id_idx})
                #     id_idx += 1
                # # body identifiers
                # for identifier in func.identifiers:
                #     id_name = func.add_fun_prefix(identifier)
                #     if id_name in self.id_to_idx:
                #         continue
                #     self.id_to_idx.update({id_name: id_idx})
                #     id_idx += 1
            self.cfg_matrix = np.full((bb_idx, bb_idx), -1, dtype=np.int8)
            self.dfg_matrix = np.full((id_idx, id_idx), -1, dtype=np.int8)
            # 0 (False) denotes no edge, 1 (True) denotes edge
            self.bb_var_matrix = np.full((bb_idx, id_idx), 0, dtype=bool)

            if build_graph:
                self.build_graphs(cfg_vocab, dfg_vocab)
            # self.pretty_print_graphs()

    def build_graphs(
        self, cfg_ty_to_idx: dict[str, int], dfg_ty_to_idx: dict[str, int]
    ):
        self.graph_built = True
        for func in self.functions:
            for bb in func.basic_blocks:
                bb_idx = self.bb_to_idx.get(func.add_fun_prefix(bb.name))
                # cfg
                terminator = bb.instructions[-1]
                if len(terminator.jumps) > 0:
                    if bb_idx is not None:
                        for jump, ty in terminator.jumps.items():
                            jump_idx = self.bb_to_idx.get(func.add_fun_prefix(jump))
                            # maybe 0
                            if jump_idx is not None:
                                self.cfg_matrix[bb_idx][jump_idx] = cfg_ty_to_idx.get(
                                    ty, -1
                                )

                for inst in bb.instructions:
                    # dfg
                    for data_flow in inst.data_flows:
                        operand_idx = self.id_to_idx.get(
                            func.add_fun_prefix(data_flow[0])
                        )
                        result_idx = self.id_to_idx.get(
                            func.add_fun_prefix(data_flow[1])
                        )
                        ty_idx = dfg_ty_to_idx.get(data_flow[2], -1)
                        if (
                            operand_idx is not None
                            and result_idx is not None
                            and ty_idx is not None
                        ):
                            self.dfg_matrix[operand_idx][result_idx] = ty_idx
                    # call graph
                    # (return_value, callee, args_list, [normal label, exception label])
                    if inst.call is not None and len(inst.call) > 2:
                        callee = inst.call[1]
                        # results value dfg idx
                        if inst.call[0] is not None:
                            result_idx = self.id_to_idx.get(
                                func.add_fun_prefix(inst.call[0])
                            )
                        else:
                            result_idx = None
                        # add to cfg
                        callee_idx = self.bb_to_idx.get(callee)
                        if callee_idx is not None:
                            self.cfg_matrix[bb_idx][callee_idx] = cfg_ty_to_idx.get(
                                "call.func", -1
                            )
                        for callee_func in self.functions:
                            if callee_func.name == callee:
                                for callee_bb in callee_func.basic_blocks:
                                    for callee_inst in callee_bb.instructions:
                                        if callee_inst.is_return:
                                            callee_bb_idx = self.bb_to_idx.get(
                                                callee_func.add_fun_prefix(
                                                    callee_bb.name
                                                )
                                            )
                                            if callee_bb_idx is not None:
                                                # call
                                                if len(inst.call) == 3:
                                                    self.cfg_matrix[callee_bb_idx][
                                                        bb_idx
                                                    ] = cfg_ty_to_idx.get(
                                                        "call.return", -1
                                                    )
                                                # invoke
                                                elif len(inst.call) == 5:
                                                    normal_bb_idx = self.bb_to_idx.get(
                                                        func.add_fun_prefix(
                                                            inst.call[3]
                                                        )
                                                    )
                                                    exception_bb_idx = (
                                                        self.bb_to_idx.get(
                                                            func.add_fun_prefix(
                                                                inst.call[4]
                                                            )
                                                        )
                                                    )
                                                    if (
                                                        normal_bb_idx is not None
                                                        and callee_inst.ret_type
                                                        == "ret"
                                                    ):
                                                        self.cfg_matrix[callee_bb_idx][
                                                            normal_bb_idx
                                                        ] = cfg_ty_to_idx.get(
                                                            "call.return", -1
                                                        )
                                                    if (
                                                        exception_bb_idx is not None
                                                        and callee_inst.ret_type
                                                        == "resume"
                                                    ):
                                                        self.cfg_matrix[callee_bb_idx][
                                                            exception_bb_idx
                                                        ] = cfg_ty_to_idx.get(
                                                            "call.resume", -1
                                                        )
                                            # add call.return to dfg
                                            if callee_inst.ret_value is not None:
                                                callee_return_idx = self.id_to_idx.get(
                                                    callee_func.add_fun_prefix(
                                                        callee_inst.ret_value
                                                    )
                                                )
                                                if (
                                                    callee_return_idx is not None
                                                    and result_idx is not None
                                                ):
                                                    self.dfg_matrix[callee_return_idx][
                                                        result_idx
                                                    ] = dfg_ty_to_idx.get(
                                                        "call.return", -1
                                                    )
                                # add call.args to dfg
                                call_arg_list = inst.call[2]
                                callee_arg_list = callee_func.args
                                for call_arg, callee_arg in zip(
                                    call_arg_list, callee_arg_list
                                ):
                                    call_arg_idx = self.id_to_idx.get(
                                        func.add_fun_prefix(call_arg)
                                    )
                                    callee_arg_idx = self.id_to_idx.get(
                                        callee_func.add_fun_prefix(callee_arg)
                                    )
                                    if (
                                        call_arg_idx is not None
                                        and callee_arg_idx is not None
                                    ):
                                        self.dfg_matrix[call_arg_idx][
                                            callee_arg_idx
                                        ] = dfg_ty_to_idx.get("call.arg", -1)
                # bb-var edges
                ids = set(bb.identifiers)
                for identifier in ids:
                    id_idx = self.id_to_idx.get(func.add_fun_prefix(identifier))
                    bb_idx = self.bb_to_idx.get(func.add_fun_prefix(bb.name))
                    if id_idx is not None and bb_idx is not None:
                        self.bb_var_matrix[bb_idx][id_idx] = True

    def pretty_print_graphs(self):
        import pandas as pd

        if not self.graph_built:
            raise ValueError("The CFG and DFG is not built yet.")

        cfg_node_names = list(self.bb_to_idx.keys())
        idx_to_cfg_ty = {value: key for key, value in self.cfg_vocab.items()}
        cfg = []
        for row in self.cfg_matrix:
            cfg.append([idx_to_cfg_ty.get(ty) for ty in row])
        cfg_df = pd.DataFrame(cfg, cfg_node_names, cfg_node_names)
        cfg_df.to_excel("cfg.xlsx")

        dfg_node_names = list(self.id_to_idx.keys())
        idx_to_dfg_ty = {value: key for key, value in self.dfg_vocab.items()}
        dfg = []
        for row in self.dfg_matrix:
            dfg.append([idx_to_dfg_ty.get(ty) for ty in row])
        dfg_df = pd.DataFrame(dfg, dfg_node_names, dfg_node_names)
        dfg_df.to_excel("dfg.xlsx")

        bb_var_df = pd.DataFrame(self.bb_var_matrix, cfg_node_names, dfg_node_names)
        bb_var_df.to_excel("bb_var.xlsx")

    def generate_bb_input_strings(self, max_num_bbs) -> list[str]:
        bb_seqs = []
        for func in self.functions:
            sig_string = f"<s> define {func.name} </s>"
            if func.args_string.strip() != "":
                for arg in func.args_string.split(","):
                    sig_string += f" {arg.strip()} </s>"
            bb_seqs.append(sig_string)
            for bb in func.basic_blocks:
                bb_string = " </s> ".join([inst.string for inst in bb.instructions])
                bb_string = "<s> " + bb_string + " </s>"
                bb_seqs.append(bb_string)
            if len(bb_seqs) >= max_num_bbs:
                break
        return bb_seqs[:max_num_bbs]


def parse_ir_file(
    content: str, cfg_vocab, dfg_vocab, build_graph=True
) -> Optional[IRFile]:
    try:
        return IRFile(content, cfg_vocab, dfg_vocab, build_graph=build_graph)
    except Exception:
        return None
