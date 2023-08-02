import requests
import json
import base64


class SharekhanAPI:
    def __init__(self, api_key):
        self.url = "https://api.sharekhan.com/skapi/"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        self.api_key = api_key
        self.request_token= None,
        self.access_token = None
        self.jwt_token = None,
        self.version_id = "1005"
        self.customer_id="3619612"
        self.secure_key="j1UUsSa2JNlAzdvpkfEMBnVQEC6ChxbM"
        self.exchange="NSE"


    def login(self,token):
            try:
                # https://api.sharekhan.com/skapi/auth/login.html?api_key=EdlLay4vZqVbVOHE6XDXFnY5xQPqMvAd&state=12345&version_id=1005
                # request_token = self.decode_base64url(token)
                self.access_token = token
                self.headers.update({"Authorization": "Bearer " + self.access_token})

            except Exception as e:
                print(f"An error occurred during login: {e}")

    def decode_base64url(self, base64url_string):
            padding = '=' * (4 - (len(base64url_string) % 4))
            base64url_string += padding
            return base64.urlsafe_b64decode(base64url_string).decode('utf-8')
    
    def is_logged_in(self):
        return self.access_token is not None

    def place_order(self, order_details):
        try:
            response = requests.post(self.url + "order", headers=self.headers, data=json.dumps(order_details))
            response.raise_for_status()
            return response.json().get('orderId')
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while placing the order: {e}")

    def modify_order(self, order_id, new_order_details):
        try:
            response = requests.put(self.url + f"order/{order_id}", headers=self.headers, data=json.dumps(new_order_details))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while modifying the order: {e}")

    def cancel_order(self, order_id):
        try:
            response = requests.delete(self.url + f"order/{order_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while cancelling the order: {e}")

    def get_real_time_feed(self, symbol):
        try:
            response = requests.get(self.url + f"realtimefeed/{symbol}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while getting the real-time feed: {e}")

    def get_net_positions(self):
        try:
            response = requests.get(self.url + "netpositions", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while getting net positions: {e}")

    def get_holdings(self):
        try:
            response = requests.get(self.url + "services/holdings/"+self.customer_id, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while getting holdings: {e}")

    def get_funds(self):
        try:
            url = self.url +"services/limitstmt/"+self.exchange+"/"+self.customer_id
            print(url)
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while getting funds: {e}")

    def get_access_token(self):
        url = "https://api.sharekhan.com/skapi/services/access/token"
        headers = {
            "Content-Type": "application/json"
        }

        data = {
           "apiKey": self.api_key,
            "requestToken": self.request_token.decode(),
        }
        if self.version_id:
            data["version_id"] = self.version_id

        print(data)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response)

        if response.status_code == 200:
            self.access_token = response.json().get('access_token')
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")

api = SharekhanAPI("EdlLay4vZqVbVOHE6XDXFnY5xQPqMvAd")
api.login("3eUXk-UCGCvLDoMcByNafm9otKMx11otewk9ljIVI6a8V0a3RfUp0juUI06ERE9kIBJgn1ydzwE=")


from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
from base64 import urlsafe_b64encode, urlsafe_b64decode

key = api.secure_key.encode('utf-8')
iv = base64.b64decode("AAAAAAAAAAAAAAAAAAAAAA==")

def encryptAPIString(plaintext):
    raw = plaintext.encode('utf-8')
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv, None, 16), default_backend()).encryptor()
    ciphertext = encryptor.update(raw) + encryptor.finalize()
    return base64UrlEncode(ciphertext + encryptor.tag)

def decryptAPIString(ciphertext):
    print(ciphertext)
    enc = base64UrlDecode(ciphertext)[:-16]
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv), default_backend()).decryptor()
    return decryptor.update(enc).decode('utf-8')

def base64UrlEncode(data):
    return urlsafe_b64encode(data).rstrip(b'=')

def base64UrlDecode(base64Url):
    padding = b'=' * (4 - (len(base64Url) % 4))
    return urlsafe_b64decode(base64Url.encode() + padding)

def manipulate_request_token(request_token):
    request_id, customer_id = request_token.split("|")
    manipulated_request_token = f"{customer_id}|{request_id}"
    return manipulated_request_token

# Decrypt the request token, manipulate it, and then encrypt it again
request_token = api.access_token
decrypted_request_token = decryptAPIString(request_token)
manipulated_request_token = manipulate_request_token(decrypted_request_token)
encrypted_request_token = encryptAPIString(manipulated_request_token)
print(manipulated_request_token)
print(encrypted_request_token)
api.request_token = encrypted_request_token

print(f"Original request token: {request_token}")
print(f"Decrypted request token: {decrypted_request_token}")
print(f"Manipulated request token: {manipulated_request_token}")
print(f"Encrypted request token: {encrypted_request_token}")

api.get_access_token()


