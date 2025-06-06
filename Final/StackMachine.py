# StackMachine.py (Interprets IR code)

import sys
import struct
from typing import List, Tuple, Any, Dict, Optional, Callable

from Error import ErrorHandler
try:
    from IRCodeGenerator import IRProgramRepresentation, IRFunctionRepresentation
except ImportError:
    # Definiciones de respaldo si el archivo se ejecuta de forma independiente
    print("Warning: Could not import IR representation classes from IRCodeGenerator. Using dummies.", file=sys.stderr)
    class IRFunctionRepresentation:
        def __init__(self, name, params, return_ir_type, is_imported=False, code=None, locals_info=None):
            self.name = name; self.params = params or []; self.return_ir_type = return_ir_type
            self.is_imported = is_imported; self.code = code or []; self.locals = locals_info or {}
    class IRProgramRepresentation:
        def __init__(self): self.globals = {}; self.functions = {}

IRInstruction = Tuple[str, ...]
IR_Type = str
StackValue = Any
StackEntry = Tuple[IR_Type, StackValue]

class StackMachine:
    def __init__(self, error_handler: ErrorHandler, memory_size_bytes: int = 1024 * 1024):
        self.error_handler: ErrorHandler = error_handler
        self.stack: List[StackEntry] = []
        self.memory: bytearray = bytearray(memory_size_bytes)
        
        self.globals: Dict[str, StackEntry] = {}
        
        # Pila de llamadas: almacena tuplas de (contador_de_programa_de_retorno, entorno_de_locales_anterior)
        self.call_stack: List[Tuple[int, Dict[str, StackEntry]]] = []
        # Entorno de variables locales de la función actual
        self.locals: Dict[str, StackEntry] = {}

        self.functions_meta: Dict[str, Dict[str, Any]] = {}
        self.program_ir: List[IRInstruction] = []
        self.pc: int = 0
        self.running: bool = False
        
        # Pila para el control de flujo estructurado (bucles)
        self.control_stack: List[Dict[str, Any]] = []

    def load_ir(self, ir_program_repr: IRProgramRepresentation):
        """Carga el código IR desde la representación del programa."""
        self.program_ir = []
        self.functions_meta = {}
        self.globals = {}
        current_pc_offset = 0

        # Inicializa las variables globales con valores por defecto
        for name, global_info in ir_program_repr.globals.items():
            ir_type = global_info["type"]
            self.globals[name] = (ir_type, self._get_default_ir_value(ir_type))

        # Almacena metadatos de las funciones y construye el código IR aplanado
        for func_name, func_repr in ir_program_repr.functions.items():
            self.functions_meta[func_name] = {
                'name': func_repr.name,
                'start_pc': current_pc_offset if not func_repr.is_imported else -1,
                'params': func_repr.params,
                'locals_info': func_repr.locals,
                'return_ir_type': func_repr.return_ir_type,
                'is_imported': func_repr.is_imported,
            }
            if not func_repr.is_imported:
                self.program_ir.extend(func_repr.code)
                current_pc_offset += len(func_repr.code)

    def run(self, entry_function_name: str = "_init"):
        """Ejecuta el código IR cargado, comenzando desde la función de entrada."""
        entry_func_meta = self.functions_meta.get(entry_function_name)
        if not entry_func_meta:
            self.error_handler.add_interpretation_error(f"Entry function '{entry_function_name}' not found.", None)
            return
        if entry_func_meta['is_imported']:
            self.error_handler.add_interpretation_error(f"Cannot directly run imported function '{entry_function_name}'.", None)
            return

        self.pc = entry_func_meta['start_pc']
        self.locals = {} # _init comienza con un entorno de locales vacío
        self.running = True

        while self.running and 0 <= self.pc < len(self.program_ir):
            instr = self.program_ir[self.pc]
            opcode = instr[0]
            args = list(instr[1:])
            
            current_instr_pc = self.pc
            self.pc += 1

            method = getattr(self, f"op_{opcode}", self.op_UNKNOWN)
            try:
                method(*args)
            except IndexError:
                self.error_handler.add_interpretation_error(f"Stack underflow during '{opcode}'.", current_instr_pc)
                self.running = False
            except TypeError as te:
                self.error_handler.add_interpretation_error(f"VM Type error or arity mismatch for '{opcode}': {te}", current_instr_pc)
                self.running = False
            except ZeroDivisionError as zde:
                self.error_handler.add_interpretation_error(f"Runtime error: {str(zde)} during '{opcode}'.", current_instr_pc)
                self.running = False
            except Exception as e:
                self.error_handler.add_interpretation_error(f"Unexpected runtime error during '{opcode}': {type(e).__name__} - {str(e)}", current_instr_pc)
                self.running = False

    def op_UNKNOWN(self, *args):
        bad_opcode = self.program_ir[self.pc-1][0]
        self.error_handler.add_interpretation_error(f"Unknown IR opcode: {bad_opcode}", self.pc - 1)
        self.running = False
    
    def _get_default_ir_value(self, ir_type: IR_Type) -> StackValue:
        if ir_type == 'I': return 0
        if ir_type == 'F': return 0.0
        return None

    def _pop_stack(self, op_name_for_error: str) -> StackEntry:
        if not self.stack: raise IndexError(f"{op_name_for_error}: Stack underflow.")
        return self.stack.pop()

    def _pop_typed_val(self, expected_ir_type: IR_Type, op_name: str) -> StackValue:
        ir_type, val = self._pop_stack(op_name)
        if ir_type != expected_ir_type:
            raise TypeError(f"{op_name}: Expected IR type '{expected_ir_type}', got '{ir_type}'.")
        return val

    # --- Operaciones de Pila y Aritmética ---
    def op_CONSTI(self, value: int): self.stack.append(('I', int(value)))
    def op_CONSTF(self, value: float): self.stack.append(('F', float(value)))

    def op_ADDI(self): r=self._pop_typed_val('I',"ADDI");l=self._pop_typed_val('I',"ADDI"); self.stack.append(('I', l + r))
    def op_SUBI(self): r=self._pop_typed_val('I',"SUBI");l=self._pop_typed_val('I',"SUBI"); self.stack.append(('I', l - r))
    def op_MULI(self): r=self._pop_typed_val('I',"MULI");l=self._pop_typed_val('I',"MULI"); self.stack.append(('I', l * r))
    def op_DIVI(self): 
        r=self._pop_typed_val('I',"DIVI");l=self._pop_typed_val('I',"DIVI")
        if r == 0: raise ZeroDivisionError("Integer division by zero")
        self.stack.append(('I', l // r))

    def op_ADDF(self): r=self._pop_typed_val('F',"ADDF");l=self._pop_typed_val('F',"ADDF"); self.stack.append(('F', l + r))
    def op_SUBF(self): r=self._pop_typed_val('F',"SUBF");l=self._pop_typed_val('F',"SUBF"); self.stack.append(('F', l - r))
    def op_MULF(self): r=self._pop_typed_val('F',"MULF");l=self._pop_typed_val('F',"MULF"); self.stack.append(('F', l * r))
    def op_DIVF(self):
        r=self._pop_typed_val('F',"DIVF");l=self._pop_typed_val('F',"DIVF")
        if r == 0.0: raise ZeroDivisionError("Float division by zero")
        self.stack.append(('F', l / r))

    def _compare_op(self, py_operator: Callable[[Any, Any], bool], ir_operand_type: IR_Type, op_name: str):
        r = self._pop_typed_val(ir_operand_type, f"{op_name}_rhs")
        l = self._pop_typed_val(ir_operand_type, f"{op_name}_lhs")
        self.stack.append(('I', 1 if py_operator(l, r) else 0))

    def op_LTI(self): self._compare_op(lambda a,b: a < b,  'I', "LTI")
    def op_LEI(self): self._compare_op(lambda a,b: a <= b, 'I', "LEI")
    def op_GTI(self): self._compare_op(lambda a,b: a > b,  'I', "GTI")
    def op_GEI(self): self._compare_op(lambda a,b: a >= b, 'I', "GEI")
    def op_EQI(self): self._compare_op(lambda a,b: a == b, 'I', "EQI")
    def op_NEI(self): self._compare_op(lambda a,b: a != b, 'I', "NEI")
    
    def op_LTF(self): self._compare_op(lambda a,b: a < b,  'F', "LTF")
    def op_LEF(self): self._compare_op(lambda a,b: a <= b, 'F', "LEF")
    def op_GTF(self): self._compare_op(lambda a,b: a > b,  'F', "GTF")
    def op_GEF(self): self._compare_op(lambda a,b: a >= b, 'F', "GEF")
    def op_EQF(self): self._compare_op(lambda a,b: a == b, 'F', "EQF")
    def op_NEF(self): self._compare_op(lambda a,b: a != b, 'F', "NEF")
    
    def op_NOTI(self): self.stack.append(('I', 1 if self._pop_typed_val('I', "NOTI") == 0 else 0))

    # CORREGIDO: Mejora de formato para PRINTI y PRINTF para que coincida con las salidas de prueba.
    def op_PRINTI(self): print(self._pop_typed_val('I', "PRINTI"), end=' ')
    def op_PRINTF(self): print(self._pop_typed_val('F', "PRINTF"), end=' ')
    def op_PRINTB(self): 
        char_code = self._pop_typed_val('I', "PRINTB")
        try: print(chr(char_code), end='')
        except ValueError: self.error_handler.add_interpretation_error(f"PRINTB: Value {char_code} invalid for chr().", self.pc-1)

    def op_ITOF(self): self.stack.append(('F', float(self._pop_typed_val('I', "ITOF"))))
    def op_FTOI(self): self.stack.append(('I', int(self._pop_typed_val('F', "FTOI"))))

    # --- Operaciones de Variables ---
    def op_LOCAL_GET(self, name: str):
        if name in self.locals: self.stack.append(self.locals[name])
        else: self.error_handler.add_interpretation_error(f"Runtime: Local variable '{name}' not found.", self.pc-1); self.running = False
    def op_LOCAL_SET(self, name: str): self.locals[name] = self._pop_stack(f"LOCAL_SET '{name}'")
        
    def op_GLOBAL_GET(self, name: str):
        if name in self.globals: self.stack.append(self.globals[name])
        else: self.error_handler.add_interpretation_error(f"Runtime: Global variable '{name}' not found.", self.pc-1); self.running = False
    def op_GLOBAL_SET(self, name: str): self.globals[name] = self._pop_stack(f"GLOBAL_SET '{name}'")

    # --- Llamadas a Funciones (CORREGIDO Y SIMPLIFICADO) ---
    def op_CALL(self, func_name: str):
        target_func_meta = self.functions_meta.get(func_name)
        if not target_func_meta:
            self.error_handler.add_interpretation_error(f"Runtime: CALL to undefined IR function '{func_name}'.", self.pc-1); self.running = False; return

        if target_func_meta.get('is_imported', False):
            if func_name == "put_image":
                h, w, b = self._pop_typed_val('I', "h"), self._pop_typed_val('I', "w"), self._pop_typed_val('I', "b")
                # En una implementación real, aquí se dibujaría la imagen. Para la prueba, es un stub.
                # print(f"[STUB] put_image(base={b}, width={w}, height={h}) called.")
                self.stack.append(('I', 0)) # Retorna un entero (0 = éxito)
                return
            else:
                 self.error_handler.add_interpretation_error(f"Runtime: CALL to unhandled imported function '{func_name}'.", self.pc-1); self.running = False; return

        # Guarda el estado actual para poder retornar
        self.call_stack.append((self.pc, self.locals))
        
        # Prepara el nuevo entorno de locales para la función llamada
        new_locals = {}
        num_params = len(target_func_meta['params'])
        # Los argumentos están en la pila, se sacan para pasarlos como parámetros
        args_from_stack = self.stack[-num_params:] if num_params > 0 else []
        self.stack = self.stack[:-num_params] if num_params > 0 else self.stack

        # Asigna argumentos a los nombres de los parámetros
        for (param_name, _), arg_val in zip(target_func_meta['params'], args_from_stack):
            new_locals[param_name] = arg_val

        # Inicializa las variables locales de la función
        for local_name, local_ir_type in target_func_meta['locals_info'].items():
            new_locals[local_name] = (local_ir_type, self._get_default_ir_value(local_ir_type))
        
        self.locals = new_locals
        self.pc = target_func_meta['start_pc']

    def op_RET(self):
        if not self.call_stack:
            self.running = False # Retorno de _init, fin del programa
            return
        # Restaura el estado de la función que llamó
        return_pc, previous_locals = self.call_stack.pop()
        self.pc = return_pc
        self.locals = previous_locals

    # --- Operaciones de Memoria ---
    def _ensure_memory_access(self, address: int, num_bytes: int, op_name: str):
        if not (0 <= address and address + num_bytes <= len(self.memory)):
            raise IndexError(f"{op_name}: Memory access out of bounds (addr={address}, len={num_bytes}, capacity={len(self.memory)}).")

    # CORREGIDO: Orden de pop en POKE*
    def op_PEEKB(self): 
        addr = self._pop_typed_val('I', "PEEKB_addr"); self._ensure_memory_access(addr, 1, "PEEKB")
        self.stack.append(('I', int(self.memory[addr])))
    def op_POKEB(self): 
        val = self._pop_typed_val('I', "POKEB_val"); addr = self._pop_typed_val('I', "POKEB_addr")
        if not (0 <= val <= 255): raise ValueError(f"POKEB value {val} not in byte range [0-255].")
        self._ensure_memory_access(addr, 1, "POKEB"); self.memory[addr] = val

    def op_PEEKI(self):
        addr = self._pop_typed_val('I', "PEEKI_addr"); self._ensure_memory_access(addr, 4, "PEEKI") 
        self.stack.append(('I', int.from_bytes(self.memory[addr:addr+4], byteorder='little', signed=True)))
    def op_POKEI(self):
        val = self._pop_typed_val('I', "POKEI_val"); addr = self._pop_typed_val('I', "POKEI_addr")
        self._ensure_memory_access(addr, 4, "POKEI"); self.memory[addr:addr+4] = val.to_bytes(4, byteorder='little', signed=True)

    def op_PEEKF(self):
        addr = self._pop_typed_val('I', "PEEKF_addr"); self._ensure_memory_access(addr, 8, "PEEKF")
        self.stack.append(('F', struct.unpack('<d', self.memory[addr:addr+8])[0]))
    def op_POKEF(self):
        val = self._pop_typed_val('F', "POKEF_val"); addr = self._pop_typed_val('I', "POKEF_addr")
        self._ensure_memory_access(addr, 8, "POKEF"); self.memory[addr:addr+8] = struct.pack('<d', val)

    # CORREGIDO: GROW devuelve la dirección base, no el nuevo tamaño.
    def op_GROW(self): 
        num_units_to_grow = self._pop_typed_val('I', "GROW_size")
        if num_units_to_grow < 0:
            raise ValueError("GROW size cannot be negative.")
        
        base_address = len(self.memory)
        self.memory.extend(bytearray(num_units_to_grow))
        self.stack.append(('I', base_address)) # Empuja la dirección base del nuevo bloque

    # --- Control de Flujo Estructurado (CORREGIDO Y SIMPLIFICADO) ---
    def _find_matching_label(self, start_op: str, end_op: str, else_op: Optional[str] = None) -> int:
        nesting_level = 1
        search_pc = self.pc
        while search_pc < len(self.program_ir):
            opcode = self.program_ir[search_pc][0]
            if opcode == start_op:
                nesting_level += 1
            elif opcode == end_op:
                nesting_level -= 1
                if nesting_level == 0:
                    return search_pc
            elif else_op and opcode == else_op:
                if nesting_level == 1:
                    return search_pc
            search_pc += 1
        raise RuntimeError(f"{start_op} at pc={self.pc-1} has no matching {end_op}")

    def op_IF(self):
        cond_val = self._pop_typed_val('I', "IF_cond")
        if cond_val == 0: # Si es falso, salta a ELSE o ENDIF
            jump_pc = self._find_matching_label('IF', 'ENDIF', 'ELSE')
            self.pc = jump_pc + 1

    def op_ELSE(self): # Salto incondicional a ENDIF
        jump_pc = self._find_matching_label('IF', 'ENDIF')
        self.pc = jump_pc + 1

    def op_ENDIF(self): pass

    def op_LOOP(self): self.control_stack.append({'type': 'LOOP', 'start_pc': self.pc - 1})

    def op_ENDLOOP(self):
        if not self.control_stack or self.control_stack[-1]['type'] != 'LOOP':
            raise RuntimeError(f"ENDLOOP at pc={self.pc-1} without matching LOOP.")
        loop_info = self.control_stack.pop()
        self.pc = loop_info['start_pc']

    def op_CBREAK(self):
        cond_val = self._pop_typed_val('I', "CBREAK_cond")
        if cond_val != 0: # Si es verdadero, romper el bucle
            if not self.control_stack or self.control_stack[-1]['type'] != 'LOOP':
                raise RuntimeError(f"CBREAK at pc={self.pc-1} not inside a LOOP.")
            
            # Encuentra el ENDLOOP que corresponde al bucle actual
            nesting_level = 1
            search_pc = self.control_stack[-1]['start_pc'] + 1
            while search_pc < len(self.program_ir):
                opcode = self.program_ir[search_pc][0]
                if opcode == 'LOOP':
                    nesting_level += 1
                elif opcode == 'ENDLOOP':
                    nesting_level -= 1
                    if nesting_level == 0:
                        self.pc = search_pc + 1
                        self.control_stack.pop() # Se saca de la pila porque se rompió el bucle
                        return
                search_pc += 1
            raise RuntimeError("Could not find matching ENDLOOP for CBREAK.")

    def op_CONTINUE(self):
        if not self.control_stack or self.control_stack[-1]['type'] != 'LOOP':
            raise RuntimeError(f"CONTINUE at pc={self.pc-1} not inside a LOOP.")
        self.pc = self.control_stack[-1]['start_pc']


# --- Main Driver ---
def main():
    import os
    from Lexer import tokenize 
    from Parser import Parser
    from SemanticAnalyzer import SemanticAnalyzer
    from IRCodeGenerator import IRCodeGenerator

    if len(sys.argv) < 2:
        print("Usage: python StackMachine.py <gox_file_to_compile_and_run>")
        sys.exit(1)

    source_file_path = sys.argv[1]
    if not (os.path.exists(source_file_path) and source_file_path.endswith('.gox')):
        print(f"Error: Source file not found or not a .gox file: '{source_file_path}'"); sys.exit(1)
    
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e: print(f"Error reading file '{source_file_path}': {e}"); sys.exit(1)

    print(f"\n--- Compiling and Running: {source_file_path} ---")
    error_handler = ErrorHandler()

    print("\n1. Lexical Analysis...")
    tokens = tokenize(content, error_handler)
    error_handler.exit_if_errors()
    print("   Success.")

    print("\n2. Parsing...")
    parser = Parser(tokens, error_handler)
    ast_root = parser.parse()
    error_handler.exit_if_errors()
    print("   Success.")

    print("\n3. Semantic Analysis...")
    semantic_analyzer = SemanticAnalyzer(error_handler)
    semantic_analyzer.analyze(ast_root)
    error_handler.exit_if_errors()
    print("   Success.")
    
    print("\n4. IR Code Generation...")
    ir_generator = IRCodeGenerator(error_handler)
    ir_generator.print_ir_to_console = False # Desactiva la impresión detallada de IR para una salida más limpia
    ir_program_representation = ir_generator.generate_ir(ast_root)
    error_handler.exit_if_errors()
    print("   Success.")

    print("\n--- Program Output ---")
    vm = StackMachine(error_handler)
    vm.load_ir(ir_program_representation)
    error_handler.exit_if_errors()
    
    vm.run(entry_function_name="_init")

    if error_handler.has_errors():
        print("\n\n--- STACK MACHINE EXECUTION ERRORS ---")
        error_handler.report_errors()
    else:
        print("\n\n--- Execution finished successfully. ---")

if __name__ == '__main__':
    main()
