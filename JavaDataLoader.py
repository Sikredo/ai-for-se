import json
import os


class JsonDataLoader:
    json_java_map = {
        "./data/original_method.json": "_original_method",
        "./data/rename_only.json": "_rename_only",
        "./data/code_structure_change_only.json": "_code_structure_change_only",
        "./data/full_transformation.json": "_full_transformation"
    }
    def __init__(self, filepath):
        self.filepath = filepath

    def load_json(self):
        """
        Loads JSON data from a file.
        """
        try:
            with open(self.filepath, 'r') as file:
                data = json.load(file)
            return data.get("data")
        except FileNotFoundError:
            print(f"Error: The file {self.filepath} does not exist.")
            return None
        except json.JSONDecodeError:
            print("Error: The file contains invalid JSON.")
            return None

    def load_java_file(self, filepath):
        """
        Loads a Java source file.
        """
        try:
            with open(filepath, 'r') as file:
                return file.readlines()
        except FileNotFoundError:
            print(f"Error: The file {filepath} does not exist.")
            return None

    def prepare_data(self, data):
        """
        Prepare the data by processing Java code lines and marking vulnerabilities.
        Assumes data in format {"data": {"Netty-1": {"loc": "11-11", "input": "Java code here"}}}
        """
        results = []
        base_dir = os.path.dirname("./")  # Assumes Java files are in the same directory as the JSON file

        for vulName, value in data.items():
            loc = value.get('loc', '')

            java_file_path = os.path.join(base_dir, "vjbench-data",vulName, f"{vulName}{self.json_java_map[self.filepath]}.java")
            code_lines = self.load_java_file(java_file_path)
            if code_lines is None:
                continue

            vulnerable_lines = self.parse_loc(loc)
            processed_lines = [
                {"code": line.strip(), "vulnerable": (index + 1 in vulnerable_lines)}
                for index, line in enumerate(code_lines) if line.strip()
            ]

            results.append({"vulName": vulName, "vulData": processed_lines})
        return results

    def parse_loc(self, loc):
        """
        Parses the 'loc' string to extract ranges of line numbers.
        Example loc: "11-11, 14-16" will return {11, 14, 15, 16}
        """
        ranges = loc.split(',')
        line_numbers = set()
        for range_ in ranges:
            start, end = map(int, range_.strip().split('-'))
            line_numbers.update(range(start, end + 1))
        return line_numbers

    def get_prepared_data(self):
        """
        Loads and prepares the JSON data.
        """
        data = self.load_json()
        if data is not None:
            return self.prepare_data(data)
        return None
