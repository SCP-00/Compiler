from Nodes_AST import *
from Sym_tab import SimbolTab
from Types import typenames

class Interpreter:
    def __init__(self):
        self.globals = {}             # variables globales
        self.memory = bytearray(1<<20)  # 1MiB de “memoria”
        self.symtab = SimbolTab("global")

    def visit_Program(self, node):
        # node.decls: lista de declaraciones (const, var, func)
        for d in node.decls:
            self.visit(d)
        # si hay func main, la invocamos:
        if "main" in self.symtab:
            return self.call_func("main", [])
    
    def visit_VarDecl(self, node):
        value = self.visit(node.init) if node.init else self.default_value(node.vtype)
        self.globals[node.name] = value
        self.symtab.add(node.name, node)

    def visit_ConstDecl(self, node):
        value = self.visit(node.value)
        self.globals[node.name] = value
        self.symtab.add(node.name, node)

    def visit_FuncDecl(self, node):
        self.symtab.add(node.name, node)

    def call_func(self, name, args):
        f = self.symtab.get(name)
        frame = {}   # nuevo ámbito de variables
        for (p, _), a in zip(f.params, args):
            frame[p] = a
        # ejecuta el cuerpo
        try:
            return self._exec_block(f.body, frame)
        except ReturnException as ret:
            return ret.value

    def _exec_block(self, stmts, frame):
        old = self.globals
        self.globals = {**old, **frame}
        for s in stmts:
            self.visit(s)
        self.globals = old

    def visit_Return(self, node):
        val = self.visit(node.value)
        raise ReturnException(val)

    def visit_If(self, node):
        cond = self.visit(node.cond)
        if cond:
            self._exec_block(node.then_block, {})
        elif node.else_block:
            self._exec_block(node.else_block, {})

    def visit_While(self, node):
        while self.visit(node.cond):
            self.visit_Block(node.body)

    def visit_Assign(self, node):
        val = self.visit(node.value)
        if node.target.is_memory:
            addr = self.visit(node.target.addr)
            self._write_mem(addr, val)
        else:
            self.globals[node.target.name] = val

    def visit_MemRead(self, node):
        addr = self.visit(node.addr)
        return self._read_mem(addr)

    def visit_BinOp(self, node):
        l = self.visit(node.left)
        r = self.visit(node.right)
        return _eval_binop(node.op, l, r)

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        return _eval_unary(node.op, v)

    def visit_Literal(self, node):
        return node.value

    def visit_Ident(self, node):
        return self.globals[node.name]

    def visit_Print(self, node):
        v = self.visit(node.expr)
        print(v)

    def _read_mem(self, addr):
        # asume 4‐bytes little endian
        return int.from_bytes(self.memory[addr:addr+4], 'little')

    def _write_mem(self, addr, val):
        self.memory[addr:addr+4] = int(val).to_bytes(4, 'little')

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# helpers de operación
def _eval_binop(op, a, b):
    return {
      '+': a+b, '-': a-b, '*': a*b, '/': a//b if isinstance(a,int) else a/b,
      '<': a<b, '<=': a<=b, '>': a>b, '>=': a>=b, '==': a==b, '!=': a!=b,
      '&&': a and b, '||': a or b
    }[op]

def _eval_unary(op, a):
    return {
      '+': +a, '-': -a, '!': not a
    }[op]