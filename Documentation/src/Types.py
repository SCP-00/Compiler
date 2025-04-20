#Types.py

typenames = { 'bool', 'char', 'float', 'int', 'string', 'void'}

#BinOps
bin_ops = {
	# Integer operations
	('int', '+', 'int') : 'int',
	('int', '-', 'int') : 'int',
	('int', '*', 'int') : 'int', 
	('int', '/', 'int') : 'int',

	('int', '<', 'int')  : 'bool',
	('int', '<=', 'int') : 'bool',
	('int', '>', 'int')  : 'bool',
	('int', '>=', 'int') : 'bool',
	('int', '==', 'int') : 'bool',
	('int', '!=', 'int') : 'bool',

	# Float operations
	('float', '+', 'float') : 'float',
	('float', '-', 'float') : 'float',
	('float', '*', 'float') : 'float',
	('float', '/', 'float') : 'float',

	('float', '<', 'float')  : 'bool',
	('float', '<=', 'float') : 'bool',
	('float', '>', 'float')  : 'bool',
	('float', '>=', 'float') : 'bool',
	('float', '==', 'float') : 'bool',
	('float', '!=', 'float') : 'bool',

	# Bools
	('bool', '&&', 'bool') : 'bool',
	('bool', '||', 'bool') : 'bool',
	('bool', '==', 'bool') : 'bool',
	('bool', '!=', 'bool') : 'bool',

	# Char
	('char', '<', 'char')  : 'bool',
	('char', '<=', 'char') : 'bool',
	('char', '>', 'char')  : 'bool',
	('char', '>=', 'char') : 'bool',
	('char', '==', 'char') : 'bool',
	('char', '!=', 'char') : 'bool',
}

def check_binop(op, left_type, right_type):
	return bin_ops.get((left_type, op, right_type))

unary_ops = {
	('+', 'int') : 'int',
	('-', 'int') : 'int',
	('^', 'int') : 'int',
    
	('+', 'float') : 'float',
	('-', 'float') : 'float',

	('!', 'bool') : 'bool',
}

def check_unaryop(op, operand_type):
	return unary_ops.get((op, operand_type))