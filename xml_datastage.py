import os
import re
import logging
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from functools import reduce

class XMLDataStageProcessor:
    def __init__(self, src_xml_dir: str, job_xml_dir: str, analyze_dir: str):
        self.src_xml_dir = src_xml_dir
        self.job_xml_dir = job_xml_dir
        self.analyze_dir = analyze_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def split_xml_file(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for job in root.findall('./Job'):
            job_identifier = job.attrib.get('Identifier')
            new_tree = ET.ElementTree(job)
            job_name = f'{job_identifier}.xml'
            new_tree.write(os.path.join(self.job_xml_dir, job_name), encoding='utf-8', xml_declaration=True)
        logging.info(f"XML has been split into {len(root.findall('./Job'))} files.")

    def load_xml(self, xml_input: str) -> ET.Element:
        try:
            if os.path.isfile(xml_input):
                # If it's a file path, use ET.parse to read the file
                tree = ET.parse(xml_input)
            else:
                # If it's not a file path, assume it's an XML string and use ET.fromstring
                tree = ET.ElementTree(ET.fromstring(xml_input))
            return tree.getroot()
        except ET.ParseError as e:
            logging.error(f"Error parsing XML: {xml_input}. Error: {e}")
            raise
        except FileNotFoundError as e:
            logging.error(f"File not found: {xml_input}. Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            raise

    def get_job_name(self, root: ET.Element) -> str:
        return root.attrib.get('Identifier')

    def get_identifier(self, root: ET.Element) -> List[str]:
        return [record.attrib['Identifier'] for record in root.findall('.//Record[@Identifier]')
                if record.attrib['Identifier'] not in ['ROOT', 'V0']]

    def get_step_name(self, root: ET.Element, record_identifiers: List[str]) -> Dict[str, str]:
        step_name = {}
        for record_id in record_identifiers:
            record = root.find(f".//Record[@Identifier='{record_id}']")
            name = record.find(".//Property[@Name='Name']").text if record is not None else None
            step_name[record_id] = name
        return step_name

    def get_summary_job(self, root: ET.Element) -> Dict[str, Dict[str, str]]:
            record = root.find(".//Record[@Identifier='V0']")
            
            stage_list = []
            stage_names = []
            stage_type_ids = []
            link_names = []
            target_stage_ids = []
            
            stage_list = record.find(".//Property[@Name='StageList']").text.split('|')
            stage_names = record.find(".//Property[@Name='StageNames']").text.split('|')
            stage_type_ids = record.find(".//Property[@Name='StageTypeIDs']").text.split('|')
            link_names = record.find(".//Property[@Name='LinkNames']").text.split('|')
            link_output_ids = record.find(".//Property[@Name='LinkSourcePinIDs']").text.split('|')
            target_stage_ids = record.find(".//Property[@Name='TargetStageIDs']").text.split('|')
            
            summary_job = {}
            for i in range(len(stage_names)):
                summary_job[stage_list[i]] = {
                    "Stage Name": stage_names[i],
                    "Stage Type ID": stage_type_ids[i],
                    "Link Output ID": link_output_ids[i],
                    "Link Output": link_names[i],
                    "Target Stage": target_stage_ids[i]
                }
                
            return summary_job

    def get_stage_inout(self, root: ET.Element, summary_job: Dict[str, Dict[str, str]], step_name: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Lấy thông tin input và output pins của các stage từ XML.

        :param root: Phần tử gốc của XML.
        :param summary_job: Từ điển chứa thông tin summary job.
        :param step_name: Từ điển ánh xạ pin IDs thành tên step.
        :return: Từ điển chứa thông tin input và output pins của các stage.
        """
        record_pin_inout = {}
        for stage_id in summary_job.keys():
            record = root.find(f".//Record[@Identifier='{stage_id}']")
            
            # Lấy thông tin input pins
            if record.find(".//Property[@Name='InputPins']") is not None:
                input_pin = record.find(".//Property[@Name='InputPins']").text.split('|')
                input_pin = [step_name.get(pin, pin) for pin in input_pin]  # Thay thế input pins bằng giá trị từ step_name
            else:
                input_pin = []
            
            # Lấy thông tin output pins
            if record.find(".//Property[@Name='OutputPins']") is not None:
                output_pin = record.find(".//Property[@Name='OutputPins']").text.split('|')
                output_pin = [step_name.get(pin, pin) for pin in output_pin]  # Thay thế output pins bằng giá trị từ step_name
            else:
                output_pin = []
            
            # Lưu thông tin vào từ điển
            record_pin_inout[stage_id] = {"input_pin": input_pin, "output_pin": output_pin}
            
        return record_pin_inout
            

    def extract_and_add_variables(self, sql_query: str, variable_list: List[str]) -> None:
        """
        Extract variables from the SQL query that start with '#' or ':' 
        and end with '#' or '.' or ','.
        Adds them to the provided list if they are not already present.

        :param sql_query: str, the SQL query string to process.
        """
        matches = re.findall(r'#\w+(?:\.\w+)?#', sql_query)
        for match in matches:
            if match not in variable_list:
                variable_list.append(match)

    def get_preformatted_query(self, root: ET.Element, record_identifier: List[str], variable_list: List[str]) -> Dict[str, str]:
        preformatted_query = {}
        for i in record_identifier:
            value = root.find(f".//Record[@Identifier='{i}']//Collection[@Name='Properties']//SubRecord/Property[@PreFormatted='1']")
            if value is not None:
                sql_query = value.text
                root_ = ET.fromstring(sql_query)
                select_statement = root_.find(".//SelectStatement")
                after_sql = root_.find(".//AfterSQL")
                if select_statement is not None:
                    preformatted_query[i] = {"Query": select_statement.text.strip(), "Usage": "Select Statement"}
                    self.extract_and_add_variables(select_statement.text.strip(), variable_list)
                elif after_sql is not None:
                    preformatted_query[i] = {"Query": after_sql.text.strip(), "Usage": "After SQL"}
                    self.extract_and_add_variables(after_sql.text.strip(), variable_list)
        return preformatted_query

    def get_column(self, root: ET.Element, record_identifier: List[str], step_name: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
        col_dict = {}
        source_column = {}
        for i in record_identifier:
            collection_col_exist = root.find(f".//Record[@Identifier='{i}']//Collection[@Name='Columns']")
            if collection_col_exist is not None:
                sub_records = collection_col_exist.findall(".//SubRecord")
                for sub_record in sub_records:
                    column_name = sub_record.find(".//Property[@Name='Name']").text
                    column_description = sub_record.find(".//Property[@Name='Description']").text if sub_record.find(".//Property[@Name='Description']") is not None else ''
                    sql_type = sub_record.find(".//Property[@Name='SqlType']").text
                    precision = sub_record.find(".//Property[@Name='Precision']").text
                    scale = sub_record.find(".//Property[@Name='Scale']").text
                    nullable = sub_record.find(".//Property[@Name='Nullable']").text
                    key_position = sub_record.find(".//Property[@Name='KeyPosition']").text
                    if step_name[i] not in col_dict:
                        col_dict[step_name[i]] = []
                    col_dict[step_name[i]].append({
                        "column_name": column_name,
                        "column_description": column_description,
                        "sql_type": int(sql_type),
                        "precision": int(precision),
                        "scale": int(scale),
                        "nullable": int(nullable),
                        "key_position": int(key_position)
                    })
                    if sub_record.find(".//Property[@Name='SourceColumn']") is not None:
                        source_column[column_name] = sub_record.find(".//Property[@Name='SourceColumn']").text  
        return col_dict

    def stage_analyze(self, file_path: str, variable_list: List[str]) -> tuple[ET.Element, Dict[str, str], Dict[str, Dict[str, str]], Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        root = self.load_xml(file_path)
        record_identifier = self.get_identifier(root)
        step_name = self.get_step_name(root, record_identifier)
        summary_job = self.get_summary_job(root)
        record_pin_inout = self.get_stage_inout(root, summary_job, step_name)
        preformatted_query = self.get_preformatted_query(root, record_identifier, variable_list)
        col_dict = self.get_column(root, record_identifier, step_name)
        for stage_id, stage_details in summary_job.items():
            input_pin = record_pin_inout.get(stage_id, {}).get("input_pin", "")
            stage_details["Link Input"] = input_pin
        return root, step_name, summary_job, col_dict, record_pin_inout, preformatted_query

    def get_properties_filter(self, root: ET.Element, stage_id: str) -> List[Dict[str, str]]:
        properties = []
        collection_prop = root.find(f".//Record[@Identifier='{stage_id}']//Collection[@Name='Properties']")
        if collection_prop:
            sub_records = collection_prop.findall(".//SubRecord")
            where_condition = sub_records[0].find(".//Property[@Name='Value']").text
            where_condition = re.sub(r"^\\\(2\)|\\\(2\)0$", "", where_condition)
            properties = where_condition.split(r"\(2)0\(1)\(3)where\(2)")
            properties = [prop for prop in properties if prop]
        return properties

    def get_exp_transformer(self, root: ET.Element, stage_id: str) -> str:
        collection_prop = root.find(f".//Record[@Identifier='{stage_id}']//Collection[@Name='MetaBag']")
        if collection_prop:
            sub_record = collection_prop.find(".//SubRecord//Property[@Name='Value'][@PreFormatted='1']")               
            if sub_record is not None:
                value_property = sub_record.text
                return value_property

    def get_properties_lookup(self, root: ET.Element, stage_id: str) -> str:
        collection_exp = root.find(f".//Record[@Identifier='{stage_id}']//Collection[@Name='MetaBag']")
        if collection_exp:
            sub_record = collection_exp.find(".//SubRecord//Property[@Name='Value'][@PreFormatted='1']")               
            if sub_record is not None:
                value_property = sub_record.text
                return value_property

    