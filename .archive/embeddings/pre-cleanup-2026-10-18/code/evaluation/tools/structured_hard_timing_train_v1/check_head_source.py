#!/usr/bin/env python3
"""Freeze artificial head fixtures and resolve globals; never open measured data."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import symtable
import sys

sys.dont_write_bytecode=True
TOOLS=Path('/embedding/code/evaluation/tools/structured_hard_timing_train_v1')

def main():
    assert Path('/.dockerenv').is_file()
    source=TOOLS/'head_fit_diagnostic.py';body=source.read_bytes()
    symbols=symtable.symtable(body.decode('utf-8'),str(source),'exec')
    known={x.get_name() for x in symbols.get_symbols()}|set(dir(builtins))|{'__file__','__name__'}
    refs=[];scopes=[]
    def inspect(table):
        scopes.append(table.get_name())
        for item in table.get_symbols():
            if item.is_referenced() and item.is_global():refs.append((table.get_name(),item.get_name()))
        for child in table.get_children():inspect(child)
    inspect(symbols)
    unresolved=sorted(set(name for _,name in refs if name not in known));assert not unresolved,unresolved
    spec=importlib.util.spec_from_file_location('head_source_fixtures',source);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    result=tool.self_test()
    assert result['archive_decodes']==0 and result['model_or_head_fits']==0
    assert source.read_bytes()==body
    result.update({'source_sha256':hashlib.sha256(body).hexdigest(),'scopes':len(scopes),
                   'global_references':len(refs),'unresolved_globals':unresolved,'measured_execution':False,
                   'head_roles_sha256':hashlib.sha256((TOOLS/'head_roles.json').read_bytes()).hexdigest()})
    target=TOOLS/'head-source-fixtures-v3.json'
    with target.open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
