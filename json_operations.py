import json
import os

# Initial list of products
products = [

]


file_path = "products.json"
def initialise_json():
    # Write the initial list of products to a JSON file
    with open(file_path, 'w') as file:
        json.dump(products, file, indent=4)


# Function to create a JSON file with initial products if it doesn't exist
def create_initial_json(file_path, products):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(products, file, indent=4)
        # print(f"Created {file_path} with initial products.")
    # else:
    #     print(f"{file_path} already exists.")

def read_json():
    # Verify the update
    create_initial_json(file_path, products)
    with open(file_path, 'r') as file:
        updated_products = json.load(file)
        # print(json.dumps(updated_products, indent=4))

    return updated_products

def add_product(file_path, new_product):
    # Function to add a new product to the JSON file
    # Read the existing products from the file

    products = read_json()

    # Add the new product to the list
    if len(products) > 0:
        new_product["id"] = products[-1]["id"] + 1
    else:
        new_product["id"] = 1
    products.append(new_product)

    # Write the updated list back to the file
    with open(file_path, 'w') as file:
        json.dump(products, file, indent=4)


# Function to remove a product from the JSON file
def remove_product(file_path, product_id):
    # Read the existing products from the file
    products = read_json()

    # Remove the product with the specified id
    products = [product for product in products if product['id'] != product_id]

    # Write the updated list back to the file
    with open(file_path, 'w') as file:
        json.dump(products, file, indent=4)

# def get_product_list_json():


# New product to add
# new_product = {
#     "id": 3,
#     "name": "Tablet",
#     "accuracy": 89.8
# }

# Add the new product to the JSON file
# add_product(file_path, new_product)
# remove_product(file_path, 1)

