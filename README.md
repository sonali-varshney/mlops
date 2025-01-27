mkdir mlops
cd mlops
python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy pandas matplotlib scikit-learn

# Run the Program
python3 app/house_price.py
