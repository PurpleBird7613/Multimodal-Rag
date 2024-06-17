
# Update package list
sudo apt update

# Install Linux packages
sudo apt install -y tesseract-ocr
sudo apt install -y libtesseract-dev
sudo apt install -y poppler-utils
sudo apt install -y libgl1-mesa-glx

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Installing Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Installing nomic-embed-text from Ollama for making Embeddings
ollama run nomic-embed-text

