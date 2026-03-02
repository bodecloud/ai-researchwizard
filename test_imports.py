
try:
    from langchain_core.schema import Document
    print('SUCCESS: langchain_core.schema import works')
except ImportError as e:
    print(f'FAILED: {e}')

