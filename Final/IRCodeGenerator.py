# IRCodeGenerator.py (Generates and prints IR, returns IR structure)

import sys
from typing import Any, Dict, Optional, List, Union, Tuple
from Nodes_AST import * 
from Error import ErrorHandler
from Types import gox_typenames # For type mapping from Semantic Analysis

# IR Instruction Format: Tuple[str, ...] e.g., ('CONSTI', 10)

# Mapping GoX types (from AST node.gox_type after semantic analysis)
# to simple IR type indicators (e.g., for choosing PEEKI vs PEEKF)
GoxTypeToIRPrimitive = {
    'int': 'I',
    'float': 'F',
    'bool': 'I',  # Bools become 0 or 1 (integer) in IR
    'char': 'I',  # Chars become their ASCII/Unicode int value in IR
    'string': 'S',# Placeholder if your IR handles strings directly, else usually addresses
    'void': 'V'
}

class IRFunctionRepresentation:
    """Holds IR code and metadata for a single function."""
    def __init__(self, name: str, params: List[Tuple[str, str]], return_ir_type: str, is_imported: bool = False):
        self.name: str = name
        self.params: List[Tuple[str, str]] = params # List of (name, ir_type)
        self.locals: Dict[str, str] = {} # Local variable name -> ir_type
        self.code: List[Tuple[str, ...]] = []
        self.return_ir_type: str = return_ir_type
        self.is_imported: bool = is_imported

    def add_local(self, name: str, ir_type: str):
        if name not in self.locals and name not in [p_name for p_name, _ in self.params]:
            self.locals[name] = ir_type

    def append(self, instruction: Tuple[str, ...]):
        self.code.append(instruction)

    def to_dict(self) -> Dict[str, Any]: # For serialization if needed
        return {
            "name": self.name,
            "params": self.params,
            "locals": self.locals,
            "code": self.code,
            "return_type": self.return_ir_type,
            "is_imported": self.is_imported
        }

class IRProgramRepresentation:
    """Holds all functions and global information for the IR program."""
    def __init__(self):
        self.globals: Dict[str, Dict[str, str]] = {} # global_name -> {"type": ir_type}
        self.functions: Dict[str, IRFunctionRepresentation] = {} # func_name -> IRFunctionRepresentation

    def add_global_var(self, name: str, ir_type: str):
        if name not in self.globals:
            self.globals[name] = {"type": ir_type}

    def add_function(self, func_repr: IRFunctionRepresentation):
        if func_repr.name not in self.functions:
            self.functions[func_repr.name] = func_repr
        else:
            # Handle redefinition error if necessary (should be caught by semantic)
            pass 
    
    def to_dict(self) -> Dict[str, Any]: # For serialization if needed
        return {
            "globals": self.globals,
            "functions": {name: func.to_dict() for name, func in self.functions.items()}
        }

class IRCodeGenerator:
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler: ErrorHandler = error_handler
        self.ir_program: Optional[IRProgramRepresentation] = None
        self.current_ir_function: Optional[IRFunctionRepresentation] = None
        self.temp_var_counter: int = 0 # For generating unique temporary names if needed by IR
        self.print_ir_to_console: bool = True # Control console output

    def _new_temp_name(self) -> str:
        self.temp_var_counter += 1
        return f"$ir_temp{self.temp_var_counter}"

    def _emit(self, *args: Any): # Changed to accept *args for tuple directly
        """Appends an instruction to the current function's code and prints it."""
        instruction = tuple(args)
        if self.current_ir_function:
            self.current_ir_function.append(instruction)
            if self.print_ir_to_console:
                # Basic print format, can be enhanced
                func_name_ctx = self.current_ir_function.name
                print(f"  IR ({func_name_ctx}): {instruction}")
        else:
            self.error_handler.add_error("IR Generation: Attempt to emit instruction outside function context.", None, error_type="INTERNAL")

    def _map_gox_type_to_ir(self, gox_type: Optional[str]) -> Optional[str]:
        if gox_type is None: return None
        return GoxTypeToIRPrimitive.get(gox_type)

    def generate_ir(self, program_node: Program) -> Optional[IRProgramRepresentation]:
        """Generates and returns the IR Program Representation, also prints IR to console."""
        self.ir_program = IRProgramRepresentation()
        self.temp_var_counter = 0

        # 1. Declare all globals and function signatures first
        for item_node in program_node.body:
            if isinstance(item_node, (VariableDecl, ConstantDecl)):
                # Global variables/constants
                # Their gox_type should be set by semantic analyzer
                ir_type = self._map_gox_type_to_ir(item_node.gox_type)
                if ir_type:
                    self.ir_program.add_global_var(item_node.name, ir_type)
                else:
                     self.error_handler.add_error(f"IR Gen: Unknown GoX type '{item_node.gox_type}' for global '{item_node.name}'.", item_node.lineno)
            
            elif isinstance(item_node, FunctionDecl):
                param_ir_tuples = []
                for p in item_node.params:
                    p_ir_type = self._map_gox_type_to_ir(p.gox_type)
                    if not p_ir_type: self.error_handler.add_error(f"IR Gen: Unknown GoX type for param '{p.name}' in func '{item_node.name}'.", p.lineno); continue
                    param_ir_tuples.append((p.name, p_ir_type))
                
                ret_ir_type = self._map_gox_type_to_ir(item_node.return_gox_type)
                if not ret_ir_type: self.error_handler.add_error(f"IR Gen: Unknown GoX return type for func '{item_node.name}'.", item_node.lineno); continue

                func_repr = IRFunctionRepresentation(item_node.name, param_ir_tuples, ret_ir_type)
                self.ir_program.add_function(func_repr)

            elif isinstance(item_node, FunctionImportDecl):
                param_ir_tuples = []
                for p in item_node.params:
                    p_ir_type = self._map_gox_type_to_ir(p.gox_type)
                    if not p_ir_type: self.error_handler.add_error(f"IR Gen: Unknown GoX type for param '{p.name}' in imported func '{item_node.func_name}'.", p.lineno); continue
                    param_ir_tuples.append((p.name, p_ir_type)) # Name might be illustrative for imported
                
                ret_ir_type = self._map_gox_type_to_ir(item_node.return_gox_type)
                if not ret_ir_type: self.error_handler.add_error(f"IR Gen: Unknown GoX return type for imported func '{item_node.func_name}'.", item_node.lineno); continue
                
                func_repr = IRFunctionRepresentation(item_node.func_name, param_ir_tuples, ret_ir_type, is_imported=True)
                self.ir_program.add_function(func_repr)

        # 2. Create and set current function to _init for global executable code
        init_params: List[Tuple[str,str]] = [] # _init takes no params
        init_return_ir_type = 'I' # _init returns int (status)
        init_func_repr = IRFunctionRepresentation("_init", init_params, init_return_ir_type)
        self.ir_program.add_function(init_func_repr) # Add to the program representation
        self.current_ir_function = init_func_repr   # Set as current for emitting global code

        if self.print_ir_to_console: print(f"IR Function: _init")

        # 3. Generate IR for global statements and initializers (into _init)
        for item_node in program_node.body:
            if not isinstance(item_node, (FunctionDecl, FunctionImportDecl)):
                self.visit(item_node) # This will emit to self.current_ir_function (_init)
        
        # Ensure _init returns
        if not self.current_ir_function.code or self.current_ir_function.code[-1][0] != 'RET':
            self._emit('CONSTI', 0)
            self._emit('RET')

        # 4. Generate IR for each user-defined function body
        for item_node in program_node.body:
            if isinstance(item_node, FunctionDecl):
                self.visit_FunctionDecl_body(item_node) # Separate method to handle body generation

        if self.print_ir_to_console: print("--- IR Generation Complete ---")
        return self.ir_program

    # --- Visitor Methods (renamed to generate_... or visit_... for clarity) ---
    def visit(self, node: Optional[Node]) -> Any: # Returns value/temp_name if expr, None if stmt
        if node is None: return
        method_name = f'visit_{node.__class__.__name__}'
        visitor_method = getattr(self, method_name, self.generic_visit_node)
        return visitor_method(node)

    def generic_visit_node(self, node: Node):
        self.error_handler.add_error(f"IRCodeGenerator has no visit method for {type(node).__name__}", node.lineno, error_type="INTERNAL")

    # Literals
    def visit_Integer(self, node: Integer): self._emit('CONSTI', node.value)
    def visit_Float(self, node: Float): self._emit('CONSTF', node.value)
    def visit_Boolean(self, node: Boolean): self._emit('CONSTI', 1 if node.value else 0)
    def visit_Char(self, node: Char): self._emit('CONSTI', ord(node.value))
    def visit_String(self, node: String): 
        # For IR, strings usually mean loading address of string literal from data segment
        # This is a complex part. For now, assume a placeholder or error.
        # Option: Create a global for the string, then load its address.
        # string_label = self._new_temp_name() # Or a more descriptive label
        # self.ir_program.add_string_literal(string_label, node.value) # If IRProgramRepr supported this
        # self._emit('LOAD_STRING_ADDR', string_label) 
        self.error_handler.add_error("IR for String literals not fully implemented.", node.lineno)


    def visit_Location(self, node: Location): # When a variable is read
        # Semantic analysis should have attached symbol information
        is_local = False
        if self.current_ir_function:
            if node.name in self.current_ir_function.locals or \
               node.name in [p_name for p_name, _ in self.current_ir_function.params]:
                is_local = True
        
        if is_local:
            self._emit('LOCAL_GET', node.name)
        elif node.name in self.ir_program.globals:
            self._emit('GLOBAL_GET', node.name)
        else:
            self.error_handler.add_error(f"IR Gen: Undefined variable '{node.name}' used.", node.lineno)
    
    def visit_VariableDecl(self, node: VariableDecl):
        # If global, it's handled by initial pass of generate_ir() for declaration.
        # Initialization code is generated when this node is visited in _init context.
        is_global_context = self.current_ir_function is not None and self.current_ir_function.name == "_init"
        
        if not is_global_context: # Local variable declaration
            var_ir_type = self._map_gox_type_to_ir(node.gox_type or node.type_spec)
            if var_ir_type:
                self.current_ir_function.add_local(node.name, var_ir_type)
            else:
                self.error_handler.add_error(f"IR Gen: Cannot map GoX type '{node.gox_type or node.type_spec}' to IR type for local var '{node.name}'.", node.lineno)
                return

        if node.value:
            self.visit(node.value) # Value is now on stack
            if is_global_context and node.name in self.ir_program.globals:
                self._emit('GLOBAL_SET', node.name)
            elif not is_global_context:
                self._emit('LOCAL_SET', node.name)
            # else: error, trying to initialize non-existent global (should be caught by semantic)

    def visit_ConstantDecl(self, node: ConstantDecl):
        # Similar to VariableDecl for initialization code.
        # Semantic phase ensures it's not reassigned.
        # IR treats it like a variable that's set once.
        self.visit(node.value) # Value on stack
        is_global_context = self.current_ir_function is not None and self.current_ir_function.name == "_init"
        if is_global_context and node.name in self.ir_program.globals:
            self._emit('GLOBAL_SET', node.name)
        elif not is_global_context: # Constant declared in a function
            const_ir_type = self._map_gox_type_to_ir(node.gox_type or node.type_spec)
            if const_ir_type: self.current_ir_function.add_local(node.name, const_ir_type)
            self._emit('LOCAL_SET', node.name)


    def visit_Assignment(self, node: Assignment):
        # Order for POKEs: value, then address on stack. POKE pops address, then value.
        # So, generate code for address_expr first, then value_expr.
        if isinstance(node.target, MemoryAddress):
            self.visit(node.target.address_expr) # Leaves address on stack
            self.visit(node.expr)                # Leaves value on stack
            
            # Determine POKE type based on value being assigned
            # (node.expr.gox_type should be set by semantic analyzer)
            value_gox_type = node.expr.gox_type 
            if value_gox_type == 'int' or value_gox_type == 'bool': self._emit('POKEI')
            elif value_gox_type == 'float': self._emit('POKEF')
            elif value_gox_type == 'char': self._emit('POKEB')
            else: self.error_handler.add_error(f"IR Gen: Cannot POKE type '{value_gox_type}' to memory.", node.lineno)
        
        elif isinstance(node.target, Location):
            self.visit(node.expr) # Value on stack
            # Determine if local or global
            is_local = False
            if self.current_ir_function:
                if node.target.name in self.current_ir_function.locals or \
                   node.target.name in [p_name for p_name, _ in self.current_ir_function.params]:
                    is_local = True
            
            if is_local:
                self._emit('LOCAL_SET', node.target.name)
            elif node.target.name in self.ir_program.globals:
                self._emit('GLOBAL_SET', node.target.name)
            else: # Should be caught by semantic analysis
                 self.error_handler.add_error(f"IR Gen: Assignment to undeclared variable '{node.target.name}'.", node.lineno)
        else:
            self.error_handler.add_error(f"IR Gen: Invalid target for assignment: {type(node.target).__name__}", node.lineno)


    def visit_Print(self, node: Print):
        self.visit(node.expr) # Value on stack
        expr_gox_type = node.expr.gox_type
        if expr_gox_type == 'int' or expr_gox_type == 'bool': self._emit('PRINTI')
        elif expr_gox_type == 'float': self._emit('PRINTF')
        elif expr_gox_type == 'char': self._emit('PRINTB')
        else: self.error_handler.add_error(f"IR Gen: Cannot print type '{expr_gox_type}'.", node.lineno)

    # Binary/Unary ops: Use the maps from your ircode.py example for opcodes
    _ir_bin_opcodes = {
        ('int', '+', 'int'): 'ADDI', ('int', '-', 'int'): 'SUBI', ('int', '*', 'int'): 'MULI', 
        ('int', '/', 'int'): 'DIVI', # ('int', '%', 'int'): 'MODI', # Assuming MODI if available
        ('int', '<', 'int'): 'LTI', ('int', '<=', 'int'): 'LEI', ('int', '>', 'int'): 'GTI', 
        ('int', '>=', 'int'): 'GEI', ('int', '==', 'int'): 'EQI', ('int', '!=', 'int'): 'NEI',
        ('float', '+', 'float'): 'ADDF', ('float', '-', 'float'): 'SUBF', ('float', '*', 'float'): 'MULF', 
        ('float', '/', 'float'): 'DIVF',
        ('float', '<', 'float'): 'LTF', ('float', '<=', 'float'): 'LEF', ('float', '>', 'float'): 'GTF', 
        ('float', '>=', 'float'): 'GEF', ('float', '==', 'float'): 'EQF', ('float', '!=', 'float'): 'NEF',
        ('bool', '&&', 'bool'): 'ANDI', ('bool', '||', 'bool'): 'ORI', # Logical ops on bools (0/1)
        ('bool', '==', 'bool'): 'EQI', ('bool', '!=', 'bool'): 'NEI',
        ('char', '<', 'char'): 'LTI', ('char', '<=', 'char'): 'LEI', ('char', '>', 'char'): 'GTI', 
        ('char', '>=', 'char'): 'GEI', ('char', '==', 'char'): 'EQI', ('char', '!=', 'char'): 'NEI',
    }
    _ir_unary_opcodes = { # (gox_op, gox_operand_type) -> list of IR instructions
        ('+', 'int'): [], ('-', 'int'): [('CONSTI', 0), ('SWAP',), ('SUBI',)], # 0-val; SWAP if SUBI is L-R
        ('!', 'bool'): [('CONSTI', 1), ('SWAP',), ('SUBI',)], # 1-val for 0/1 bools, or specific NOTI
        ('+', 'float'): [], ('-', 'float'): [('CONSTF', 0.0), ('SWAP',), ('SUBF',)],
    }

    def visit_BinOp(self, node: BinOp):
        # For && and ||, IR needs conditional jumps for short-circuiting
        # Example for 'A && B': eval A; IF_FALSE_JUMP label_false; eval B; label_false:
        if node.op == '&&':
            # This is a placeholder, proper short-circuiting needs labels and jumps
            self.visit(node.left); self.visit(node.right)
            self._emit('ANDI') # Simple non-short-circuiting AND for bools (0/1)
            return
        elif node.op == '||':
            self.visit(node.left); self.visit(node.right)
            self._emit('ORI') # Simple non-short-circuiting OR
            return

        self.visit(node.left)
        self.visit(node.right)
        # Semantic analysis should ensure types are compatible and set node.left.gox_type etc.
        op_key = (node.left.gox_type, node.op, node.right.gox_type)
        ir_opcode = self._ir_bin_opcodes.get(op_key)
        if ir_opcode:
            self._emit(ir_opcode)
        else:
            self.error_handler.add_error(f"IR Gen: No IR for BinOp {op_key}", node.lineno)

    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.operand)
        op_key = (node.op, node.operand.gox_type)
        ir_sequence = self._ir_unary_opcodes.get(op_key)
        if ir_sequence is not None: # Allow empty list for no-op
            for instr_parts in ir_sequence:
                self._emit(*instr_parts) # Unpack tuple if it's already formed
        # MEM_ALLOC is handled by visit_MemoryAllocation
        elif node.op == '^': 
             self.error_handler.add_error("IR Gen: '^' should be MemoryAllocationNode", node.lineno)
        else:
            self.error_handler.add_error(f"IR Gen: No IR for UnaryOp {op_key}", node.lineno)
            
    def visit_MemoryAllocation(self, node: MemoryAllocation): # ^expr
        self.visit(node.size_expr) # Size on stack
        self._emit('GROW')         # Consumes size, pushes base_address (or new total size per IR spec)

    def visit_MemoryAddress(self, node: MemoryAddress): # `expr (R-value context)
        self.visit(node.address_expr) # Address on stack
        # The type of data to PEEK depends on context, should be set in node.gox_type by semantic pass
        peek_type = node.gox_type
        if peek_type == 'int' or peek_type == 'bool': self._emit('PEEKI')
        elif peek_type == 'float': self._emit('PEEKF')
        elif peek_type == 'char': self._emit('PEEKB')
        else: # Default or error
            self.error_handler.add_error(f"IR Gen: Cannot PEEK for unknown/unspecified type '{peek_type}' from memory.", node.lineno)
            self._emit('PEEKI') # Fallback to PEEKI

    def visit_TypeCast(self, node: TypeCast):
        self.visit(node.expr) # Value to cast is on stack
        from_gox = node.expr.gox_type
        to_gox = node.target_gox_type
        if from_gox == to_gox: return # No IR needed

        if from_gox == 'int' and to_gox == 'float': self._emit('ITOF')
        elif from_gox == 'float' and to_gox == 'int': self._emit('FTOI')
        elif (from_gox == 'char' and to_gox == 'int') or \
             (from_gox == 'int' and to_gox == 'char'):
            pass # Values are already compatible 'I' type in IR
        # Add other casts if supported by your IR
        else:
            self.error_handler.add_error(f"IR Gen: Unsupported type cast IR from {from_gox} to {to_gox}.", node.lineno)

    def visit_If(self, node: If):
        self.visit(node.test) # Condition (0 or 1) on stack
        self._emit('IF')      # VM pops; if 0, jumps to ELSE or ENDIF
        for stmt in node.consequence: self.visit(stmt)
        if node.alternative:
            self._emit('ELSE')
            for stmt in node.alternative: self.visit(stmt)
        self._emit('ENDIF')

    def visit_While(self, node: While):
        self._emit('LOOP')
        self.visit(node.test) # Condition (0 or 1) on stack
        # CBREAK breaks if stack top is true (non-zero). We want to loop while true.
        # So, if condition is false (0), we want to break.
        # Option 1: emit NOTI if condition is false (0 -> 1, 1 -> 0), then CBREAK
        self._emit('NOTI') # Invert condition for CBREAK
        self._emit('CBREAK')
        for stmt in node.body: self.visit(stmt)
        self._emit('ENDLOOP')

    def visit_Break(self, node: Break):
        self._emit('CONSTI', 1) # Push true for CBREAK
        self._emit('CBREAK')
    def visit_Continue(self, node: Continue): self._emit('CONTINUE')

    def visit_Return(self, node: Return):
        if node.expr:
            self.visit(node.expr) # Result on stack
        # If void func and no expr, IR RET handles it.
        self._emit('RET')

    def visit_FunctionDecl_body(self, node: FunctionDecl): # Separated for clarity
        """Generates IR for the body of a user-defined function."""
        original_func_ctx = self.current_ir_function
        # Retrieve the IRFunctionRepresentation created in the first pass
        self.current_ir_function = self.ir_program.functions.get(node.name)
        if not self.current_ir_function or self.current_ir_function.is_imported:
            self.error_handler.add_error(f"IR Gen: Could not find or tried to generate body for imported/missing func '{node.name}'.", node.lineno, error_type="INTERNAL")
            self.current_ir_function = original_func_ctx # Restore
            return
        
        if self.print_ir_to_console: print(f"IR Function: {node.name}")
        self.temp_var_counter = 0 # Reset temps for each function

        # Parameters are on stack/in frame by CALL. Locals declared in body.
        for stmt_node in node.body:
            self.visit(stmt_node)

        # Ensure RET for non-void if not explicit (Semantic should catch this)
        if not self.current_ir_function.code or self.current_ir_function.code[-1][0] != 'RET':
            if self.current_ir_function.return_ir_type == 'V': # Void
                self._emit('RET')
            else: # Non-void, semantic error. Emit default return to make IR somewhat valid.
                # self.error_handler.add_error(f"IR Gen: Non-void func '{node.name}' missing return.", node.lineno)
                if self.current_ir_function.return_ir_type == 'I': self._emit('CONSTI', 0)
                elif self.current_ir_function.return_ir_type == 'F': self._emit('CONSTF', 0.0)
                self._emit('RET')
        
        self.current_ir_function = original_func_ctx # Restore outer function context

    def visit_FunctionImportDecl(self, node: FunctionImportDecl):
        # Signatures already processed. No body to generate.
        pass
        
    def visit_FunctionCall(self, node: FunctionCall):
        for arg_node in node.args: # Evaluate args left-to-right
            self.visit(arg_node)   # Results are pushed onto stack
        self._emit('CALL', node.name)
        # Result (if any) is left on stack by CALL/RET mechanism

# --- Main Driver ---
def main():
    import os
    from Lexer import tokenize
    from Parser import Parser
    from SemanticAnalyzer import SemanticAnalyzer 
    
    if len(sys.argv) != 2:
        print("Usage: python IRCodeGenerator.py <file.gox>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not (os.path.exists(file_path) and file_path.endswith('.gox')):
        print(f"Error: File not found or not a .gox file: {file_path}"); sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e: print(f"Error reading file '{file_path}': {str(e)}"); sys.exit(1)

    print(f"\n--- Generating IR for: {file_path} ---")
    error_handler = ErrorHandler()

    tokens = tokenize(content, error_handler)
    if error_handler.has_errors(): error_handler.report_errors(); sys.exit(1)

    parser = Parser(tokens, error_handler)
    ast_root = parser.parse()
    if error_handler.has_errors() or not ast_root:
        if not ast_root: print("Parser failed to produce an AST.")
        error_handler.report_errors(); sys.exit(1)

    semantic_analyzer = SemanticAnalyzer(error_handler) # Assumes SemanticAnalyzer is correct
    semantic_analyzer.analyze(ast_root) 
    if error_handler.has_errors():
        print("\nSEMANTIC ERRORS found:"); error_handler.report_errors(); sys.exit(1)
    
    print("Lexical, Parsing, and Semantic Analysis successful.")
    print("\n--- Generated IR Code (Printed to Console) ---")

    ir_generator = IRCodeGenerator(error_handler)
    ir_program_repr = ir_generator.generate_ir(ast_root) # This now prints IR as it generates

    if error_handler.has_errors():
        print("\nIR GENERATION ERRORS:"); error_handler.report_errors(); sys.exit(1)
    
    if ir_program_repr:
        # The IR was already printed. Here you could serialize ir_program_repr to a file if needed.
        # For example, using json.dump(ir_program_repr.to_dict(), f)
        # For now, the console output is the primary IR "file".
        ir_output_file_path = file_path.replace(".gox", ".ir.json")
        try:
            with open(ir_output_file_path, 'w') as f_json:
                import json
                json.dump(ir_program_repr.to_dict(), f_json, indent=2)
            print(f"\nIR Representation also saved to: {ir_output_file_path}")
        except Exception as e:
            print(f"\nCould not save IR representation to JSON: {e}")
        print("\nIR Code generation successful.")
    else:
        print("IR Code generation failed to produce a module.")

if __name__ == "__main__":
    from Lexer import tokenize
    from Parser import Parser
    from SemanticAnalyzer import SemanticAnalyzer
    main()