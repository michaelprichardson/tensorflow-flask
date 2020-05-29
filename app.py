from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_socketio import SocketIO

import tensorflow as tf
from tensorflow import keras
import numpy as np

import os

TRAIN = False

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)

# Load the model
checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

if TRAIN:
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    model.fit(train_images, train_labels, epochs=20, callbacks=[cp_callback])
else:
    model.load_weights(checkpoint_path)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Product Class/Model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column(db.String(200))
    price = db.Column(db.Float)
    qty = db.Column(db.Integer)

    def __init__(self, name, description, price, qty):
        self.name = name
        self.description = description
        self.price = price
        self.qty = qty

# Product Schema
class ProductSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'description', 'price', 'qty')

# Init schema
product_schema = ProductSchema()
products_schema = ProductSchema(many=True)

# Create a Product
@app.route('/product', methods=['POST'])
def add_product():
    name = request.form['name']
    description = request.form['description']
    price = request.form['price']
    qty = request.form['qty']

    new_product = Product(name, description, price, qty)

    db.session.add(new_product)
    db.session.commit()

    return product_schema.jsonify(new_product)

# Get All Products
@app.route('/products', methods=['GET'])
def get_products():
    all_products = Product.query.all()
    result = products_schema.dump(all_products)
    return jsonify(result)

# Get Single Products
@app.route('/product/<id>', methods=['GET'])
def get_product(id):
    product = Product.query.get(id)
    return product_schema.jsonify(product)

# Update a Product
@app.route('/product/<id>', methods=['PUT'])
def update_product(id):
    product = Product.query.get(id)

    name = request.form['name']
    description = request.form['description']
    price = request.form['price']
    qty = request.form['qty']

    product.name = name
    product.description = description
    product.price = price
    product.qty = qty

    db.session.commit()

    return product_schema.jsonify(product)

# Delete Product
@app.route('/product/<id>', methods=['DELETE'])
def delete_product(id):
    product = Product.query.get(id)
    db.session.delete(product)
    db.session.commit()

    return product_schema.jsonify(product)


# Load the model
@app.route('/prediction', methods=['POST'])
def model_load():
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    preds = make_predictions()
    results = []

    for x in range(0, len(preds)):
        idx = np.argmax(preds[x])
        results.append(class_names[idx])
        
    return jsonify({ 'n_predictions': len(preds), 'predictions': results })


@app.route('/')
def hello():
    return jsonify({ 'message': 'Hello, World!'})

def make_predictions():
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return probability_model.predict(test_images)
    

# Run Server
if __name__ == '__main__':
    app.run(debug=True, threaded=True)