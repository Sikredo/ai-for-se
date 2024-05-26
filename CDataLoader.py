import json
import os


class CDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_json(self):
        """
        Loads JSON data from a file.
        """
        try:
            with open(self.filepath, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.filepath} does not exist.")
            return None
        except json.JSONDecodeError:
            print("Error: The file contains invalid JSON.")
            return None

    def prepare_data(self, data):
        """
        Prepare the data by marking vulnerable lines according to the 'flaw_line_no' field.
        """
        results = []
        for item in data:
            code_lines = item["code"].split("\n")
            vul_lines = set(item["flaw_line_no"])
            processed_lines = [
                {"code": line.strip(), "vulnerable": (index + 1 in vul_lines)}
                for index, line in enumerate(code_lines) if line.strip()
            ]

            results.append({
                "vulName": f"BigVul-{item['bigvul_id']}",
                "vulData": processed_lines,
                "isVulnerable": bool(item["vul"])
            })
        return results

    def get_prepared_data(self):
        """
        Loads and prepares the JSON data.
        """
        data = self.load_json()
        if data is not None:
            return self.prepare_data(data)
        return None
