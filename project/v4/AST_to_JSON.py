# AST_to_JSON.py
import json
from Nodes_AST import *

def ast_to_json(node):
    """Convierte un nodo AST a un diccionario JSON serializable."""
    if node is None:
        return None
    if isinstance(node, list):
        return [ast_to_json(item) for item in node]

    data = {"type": node.__class__.__name__}
    if isinstance(node, Program):
        data["statements"] = ast_to_json(node.statements)
    elif isinstance(node, Integer):
        data["value"] = node.value
    elif isinstance(node, Float):
        data["value"] = node.value
    elif isinstance(node, Boolean):
        data["value"] = node.value
    elif isinstance(node, String):
        data["value"] = node.value
    elif isinstance(node, Char):
        data["value"] = node.value
    elif isinstance(node, BinOp):
        data["operator"] = node.op
        data["left"] = ast_to_json(node.left)
        data["right"] = ast_to_json(node.right)
    elif isinstance(node, UnaryOp):
        data["operator"] = node.op
        data["operand"] = ast_to_json(node.operand)
    elif isinstance(node, Location):
        data["name"] = node.name
    elif isinstance(node, FunctionCall):
        data["name"] = node.name
        data["arguments"] = ast_to_json(node.args)
    elif isinstance(node, Print):
        data["expression"] = ast_to_json(node.expr)
    elif isinstance(node, Assignment):
        data["target"] = ast_to_json(node.location)
        data["value"] = ast_to_json(node.expr)
    elif isinstance(node, If):
        data["condition"] = ast_to_json(node.test)
        data["consequence"] = ast_to_json(node.consequence)
        data["alternative"] = ast_to_json(node.alternative)
    elif isinstance(node, While):
        data["condition"] = ast_to_json(node.test)
        data["body"] = ast_to_json(node.body)
    elif isinstance(node, VariableDecl):
        data["name"] = node.name
        data["type"] = node.var_type
        data["initial_value"] = ast_to_json(node.value)
    elif isinstance(node, ConstantDecl):
        data["name"] = node.name
        data["value"] = ast_to_json(node.value)
    elif isinstance(node, FunctionDecl):
        data["name"] = node.name
        data["parameters"] = ast_to_json(node.params)
        data["return_type"] = node.return_type
        data["body"] = ast_to_json(node.body)
    elif isinstance(node, Return):
        data["value"] = ast_to_json(node.expr)
    elif isinstance(node, Parameter):
        data["name"] = node.name
        data["type"] = node.param_type
    elif isinstance(node, ImportDecl):
        data["module_name"] = node.module_name
    elif isinstance(node, FunctionImportDecl):
        data["module_name"] = node.module_name
        data["params"] = ast_to_json(node.params)
        data["return_type"] = node.return_type
    elif isinstance(node, Dereference):
        data["location"] = ast_to_json(node.location)
    elif isinstance(node, Break):
        pass
    elif isinstance(node, Continue):
        pass
    return data

def save_ast_to_json(ast, filename="ast_output.json"):
    """Guarda el AST serializado en un archivo JSON."""
    ast_json = ast_to_json(ast)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ast_json, f, indent=2, ensure_ascii=False)
    return ast_json
