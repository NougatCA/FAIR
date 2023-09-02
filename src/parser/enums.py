
opcode_to_meta = {
    # unary operations
    "fneg": ("fneg", 1, False),
    # binary operations
    "add": ("add", 2, False),
    "fadd": ("add", 2, False),
    "sub": ("sub", 2, True),
    "fsub": ("sub", 2, True),
    "mul": ("mul", 2, False),
    "fmul": ("mul", 2, False),
    "udiv": ("div", 2, True),
    "sdiv": ("div", 2, True),
    "fdiv": ("div", 2, True),
    "urem": ("rem", 2, True),
    "srem": ("rem", 2, True),
    "frem": ("rem", 2, True),
    # bitwise binary operations
    "shl": ("shl", 2, True),
    "lshr": ("lshr", 2, True),
    "ashr": ("ashr", 2, True),
    "and": ("and", 2, False),
    "or": ("or", 2, False),
    "xor": ("xor", 2, False),
    # vector operations
    "extractelement": ("extractelemet", 2, True),
    "insertelement": ("insertelement", 2, True),
    "shufflevector": ("shufflevector", -1, True, ["v1", "v2", "mask"]),
    # aggregate operations
    "extractvalue": ("extractvalue", -1, True, ["value", "idx"]),
    "insertvalue": ("insertvalue", -1, True, ["value", "elt", "idx"]),
    # memory access and addressing operations
    # alloca
    "load": ("load", 1, False),
    # store
    # fence
    # cmpxchg
    # atomicrmw
    "getelementptr": ("getelementptr", -1, True, ["ptrval", "idx"]),
    # conversions operations
    "trunc": ("conversion", 1, False),
    "zext": ("conversion", 1, False),
    "sext": ("conversion", 1, False),
    "fptrunc": ("conversion", 1, False),
    "fpext": ("conversion", 1, False),
    "fptoui": ("conversion", 1, False),
    "fptosi": ("conversion", 1, False),
    "uitofp": ("conversion", 1, False),
    "sitofp": ("conversion", 1, False),
    "ptrtoint": ("conversion", 1, False),
    "inttoptr": ("conversion", 1, False),
    "bitcast": ("conversion", 1, False),
    "addrspacecast": ("conversion", 1, False),
    # other operations
    # icmp
    # fcmp
    "phi": ("phi", -1, False),
    "select": ("select", 2, True),
    "freeze": ("freeze", 1, False),
    # call
    # va_arg
    # landingpad
    # catchpad
    # cleanuppad
}
