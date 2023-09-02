import re
from typing import Optional

from .LLVMIRLexer import LLVMIRLexer, InputStream


def extract_func_defs(llvm_ir: str) -> list[str]:
    func_strings = []
    matches = re.findall(
        r"^define\s+.*?@\w+\s*\(.*?\)\s*.+?(?=^define|\Z)", llvm_ir, flags=re.M | re.S
    )
    for match in matches:
        num_open_brackets = 0
        for idx in range(len(match)):
            char = match[idx]
            if char == "{":
                num_open_brackets += 1
            elif char == "}":
                num_open_brackets -= 1
                if num_open_brackets == 0:
                    func_strings.append(match[: idx + 1])
                    break
    return func_strings


def clean_ir(content: str) -> Optional[str]:
    # the pattern of function definition
    # def_pattern = re.compile(r"^define\s+.*?@\w+\s*\(.*?\)\s*.+?(?=^define|\Z)", flags=re.S)
    # defs = def_pattern.findall(content)
    # content = re.sub(r"[%@]\S+", lambda match: match.group()[:20], content)
    func_defs = extract_func_defs(content)
    func_defs = func_defs[:20]
    if len(func_defs) == 0:
        return None
    content = "\n\n".join(func_defs)
    if len(content.split("\n")) < 5:
        return None

    # remove some tokens
    content = re.sub(r",* align \d+", " ", content)  # ", align 4" or " align 4"
    content = re.sub(
        r"(, )?![\w.]+(\(.*?\))?", "", content
    )  # !dbg, !llvm.loop, !23
    # content = re.sub(r"![\w\.]+", "", content)                   # !dbg, !llvm.loop, !23
    content = re.sub(r",* !dbg !\d+", "", content)  # ", !dbg !4" or " !dbg !3"
    content = re.sub(
        r", metadata !\w+\(?\)?", "", content
    )  # ", metadata !12" or ", metadata !XXX()"
    content = re.sub(r"metadata ", "", content)  # " metadata"
    content = re.sub(r" *;.*$", " ", content, flags=re.M)  # comments
    content = re.sub(r" #\d+", " ", content)  # attributes, e.g., "#1"
    content = re.sub(r", !llvm.loop !\d+", " ", content)  # ", !llvm.loop !6"
    content = re.sub(r"\([^(]*?@\.str.*?\)", " ", content)
    # remove other unuseful tokens
    content = re.sub(
        r"(dso_local|noundef|nonnull|nounwind|local_unnamed_addr|"
        r"getelementptr|inbounds|unnamed_addr|comdat|"
        r"dereferenceable\(\d+\))",
        " ",
        content,
    )
    return content


def tokenize_ir(string: str) -> list[str]:
    stream = InputStream(string)
    return [token.text for token in LLVMIRLexer(stream).getAllTokens()]


def find_all_values(string: str) -> list[str]:
    # true, false
    # integers
    # floats
    # null
    # none
    # undef
    # identifier
    string = " ".join(tokenize_ir(string))
    # remove some misleading types
    string = re.sub(r"\[.+?x.+?]", "", string)
    string = re.sub(r"<.+?x.+?>", "", string)
    string = re.sub(r"<?\{.+?}>?", "", string)

    matches = re.findall(
        r"\s(true|false|null|none|undef|[+-]?([0-9]*[.])?[0-9]+|[%@][-a-zA-Z$._0-9]+)",
        string,
    )
    return [match[0].strip() for match in matches]


def find_all_identifiers(string: str) -> list[str]:
    return re.findall(r"[%@][-a-zA-Z$._0-9]+", string)
