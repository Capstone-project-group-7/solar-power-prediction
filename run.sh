
#!/bin/sh

echo "Create an environment...."
python3 -m venv eflask
echo "Done!"

echo "Activating environment....."
. eflask/bin/activate
echo "Done!"

echo "Installing all required dependencies....."
pip install flask

pip install pandas

pip install numpy

pip install sklearn

pip install gunicorn

pip install flask-cors

pip install keras

pip install tensorflow-cpu

echo "Done!"

echo "Starting server..."
gunicorn wsgi:app
