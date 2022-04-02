from flask import Flask, request
import json
import yaml
import os

app = Flask(__name__)


class Data:
    __backup_file_name = "backup.yaml"
    data = {}

    def __init__(self):
        pass

    def update_data(self, data_type, data):
        self.data[data_type] = data

    def get_data(self, data_type):
        if data_type in self.data:
            return {"data": self.data[data_type]}
        return {"data": "NULL"}

    def save(self):
        yaml.dump(self.data, open(self.__backup_file_name, "w"), yaml.Dumper)

    def load(self):
        if os.path.exists(self.__backup_file_name):
            self.data = yaml.load(open(self.__backup_file_name))


my_data = Data()


@app.route("/detect_result", methods=["POST"])
def receive_detect_result():
    params = request.data.decode()
    my_data.update_data("detect", params)
    print(params, len(params))
    # p = json.loads(params)["outputs"]
    # count = {}
    # for obj in p:
    #     if not obj['label'] in count:
    #         count[obj['label']] = 1
    #     else:
    #         count[obj['label']] += 1
    #
    # print("total:", len(p), count)
    return ""


@app.route("/update_result", methods=["POST"])
def update_result():
    data = request.args.to_dict()
    if "type" in data and "data" in data:
        my_data.update_data(data["type"], data["data"])
        my_data.save()
        return True
    return False


@app.route("/get_result", methods=["POST"])
def send_result():
    data = request.args.to_dict()
    if "type" in data:
        return my_data.get_data(data["type"])


if __name__ == '__main__':
    my_data.load()
    app.run(host="0.0.0.0", port=12345)