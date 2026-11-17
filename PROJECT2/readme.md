# Setting up the environment 
````

```bash
source modules.sh
python3 -m venv .venv
source .venv/bin/activate
pip intsall -r ./requirements.txt
pip install -e .
```

Now you can call 
```python3
from sequential.blocks import CartDecomposition
```
