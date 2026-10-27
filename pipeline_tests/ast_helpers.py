"""Load real source functions without invoking optional network dependencies."""
import ast
import numpy as np
import pandas as pd


def functions(path):
    tree=ast.parse(path.read_text('utf-8-sig'));nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Assign,ast.AnnAssign)):
            nodes.append(node)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            roots=[item.name.split('.')[0] for item in node.names] if isinstance(node,ast.Import) else [(node.module or '').split('.')[0]]
            if not set(roots)&{'exchange_calendars','yfinance','collect_sec_fundamentals'}:
                nodes.append(node)
    namespace={'__file__':str(path),'__name__':'source_under_test','np':np,'pd':pd}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
