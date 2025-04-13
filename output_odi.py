import pandas as pd
import os
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import logging
import oracledb
import sys
import re
import sqlparse
from functools import reduce
import sqlglot


def create_result_file(folder_path):
    files = os.listdir(folder_path)
    
    # List to store each DataFrame
    dfs = []

    # Loop through the files and read them into DataFrames
    for file in files:
        df1 = pd.read_excel(os.path.join(folder_path, file), header=0)
        dfs.append(df1)  # Append the DataFrame to the list

    # Concatenate all DataFrames into one
    df = pd.concat(dfs, ignore_index=True)

    file_result_path = r'./extract_datastage/results.xlsx'
    # print to excel
    df.to_excel(file_result_path, index=False)
    
    return file_result_path
    
def create_dataframe(file_result_path):
    df = pd.read_excel(file_result_path, header=0)
    return df

    
def get_mapping(df: pd.DataFrame):
    """
    Function to create a mapping from a DataFrame of stage info, shortening names for source, target, and stage names.
    
    :param stage_info: A DataFrame containing information about the stages.
    :return: A dictionary with shortened stage names and corresponding source and target stages.
    """
    # Group by 'Job Name' and get unique 'Stage Name'
    df_ = df.groupby('Job Name')['Stage Name'].unique().apply(list).reset_index()
    sn_dict = df_.set_index('Job Name')['Stage Name'].to_dict()
    
    mapping = {}

    for job_name, stage_names in sn_dict.items():
        # Create a dict with key as job_name
        mapping[job_name] = {}
        
        for stage_name in stage_names:
            # Get source_stage and target_stage from DataFrame
            source_stage = df.loc[(df['Job Name'] == job_name) & (df['Stage Name'] == stage_name), 'Source Stage'].values
            target_stage = df.loc[(df['Job Name'] == job_name) & (df['Stage Name'] == stage_name), 'Target Stage'].values
            
            # Replace NaN with empty string for source_stage and target_stage
            source_stage = ["" if pd.isna(x) else x for x in source_stage.tolist()]
            target_stage = ["" if pd.isna(x) else x for x in target_stage.tolist()]

            # Append the source and target stages to the mapping dictionary
            if stage_name not in mapping[job_name]:
                mapping[job_name][stage_name] = {
                    'source_stage': source_stage,
                    'target_stage': []
                }

            # If target_stage is not in target_stage, append it
            for tgt in target_stage:
                if tgt not in mapping[job_name][stage_name]['target_stage']:
                    mapping[job_name][stage_name]['target_stage'].append(tgt)
    
    return mapping


def get_mapping_in_out(mapping):
    
    mapping_in_out = {}

    # Iterate through the mapping dictionary
    for mapping_name, stages in mapping.items():
        mapping_in_out[mapping_name] = {}  # Initialize a dictionary for each mapping_name
        source_stages = []
        target_stages = []
        comp_stages = []

        # Iterate through the stages and extract the source and target stages
        for stage_name, stage_info in stages.items():
            # Check if source_stage is equal to ['']
            if stage_info['target_stage'] == ['']:
                target_stages.append(stage_name.upper())  # Add the stage_name to the target_stages list
            if stage_info['source_stage'] == ['']:
                source_stages.append(stage_name.upper())  # Add the stage_name to the source_stages list
            # Check if target_stage is not equal to ['']
            if stage_info['target_stage'] != [''] and stage_info['source_stage'] != ['']:
                comp_stages.append(stage_name.upper())

        # convert source_stages to a tuple
        target_stage = target_stages[0]
        source_stages_tuple = [{source: target_stage} for source in source_stages]
                
        target_stages_tuple = [{target: target_stage} for target in target_stages]
        
        comp_stages = [{comp: 'SET'} for comp in comp_stages]
        combined_stages = source_stages_tuple + target_stages_tuple + comp_stages
        combined_stages_dict = {k: v for d in combined_stages for k, v in d.items()}
        
        

        # Store the source and target stages under their respective keys

        mapping_in_out[mapping_name]['comp_name'] = combined_stages_dict
    

    return mapping_in_out


def generate_response_from_text_azure(prompt_text: str, model: str = "gpt-4o", temperature: float = 0, max_tokens: int = 8000) -> str:
    """
    Generates a response from Azure OpenAI based on the given prompt text.
    
    Args
        prompt_text (str): The text prompt to send to the model.
        model (str, optional): The model to use for generating the response. Defaults to "gpt-4".
        temperature (float, optional): Sampling temperature. Defaults to 0.2.
        max_tokens (int, optional): Maximum number of tokens. Defaults to 8000.
    
    Returns:
        str: The generated response.
    """
    try:
        client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
                            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                            api_version=os.getenv("API_VERSION"))
        messages = [{"role": "system", "content": "You are an expert at SQL in IBM DataStage and ORACLE."},
                    {"role": "user", "content": prompt_text}]
        
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=1)
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error: {e}"

def read_file(file_path: str) -> str:
    """
    Reads a file and returns its content as a string.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        str: The content of the file, or an error message if there's an error.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return f"Error: File does not exist - {file_path}"
        
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return ""

def apply_prompt(prompt: str = "ok", more_rules: str = ".") -> str:
    """
    Applies the prompt to generate a response.
    
    Args:
        prompt (str): The main prompt text.
        var (str): Additional variable text.
        more_rules (str): Additional rules text.
    
    Returns:
        str: The generated response.
    """
    prompt_text = f"{prompt}\n{more_rules}"
    return generate_response_from_text_azure(prompt_text)

def format_query(query):
    query = str(query)
    return query.replace("#CONNECTION_GSTX.GSTX_DBSCHEMA#.", "")\
                .replace("#STATICPARM.CURDTDB2_DBSCHEMA#.", "")\
                .replace("#DYNAMICPARM.DATADT#", "'20231004'").replace("SFDB2_DBSCHEMA.", "")\
                .replace("#STATICPARM.SFDB2_DBSCHEMA#.", '')\
                .replace("#GSTX_LIMIT_CIF_OPEN_DT#", "20240101")\
                .replace("CURRENTDATA.", "")

def get_corrected_query(df: pd.DataFrame, connection: oracledb, more_rule_query_checker) -> dict:    
    for index, row in df.iterrows():
        if str(row['Query']) != 'nan' and row['Stage Type ID'] == 'DB2ConnectorPX':
            print(index)
            converted_sql = apply_prompt(format_query(row['Query']), more_rule_query_checker)
            # thêm vào df 
            df.loc[index, 'Query'] = converted_sql
            
            with connection.cursor() as cursor:
                try:
                    cursor.execute(format_query(converted_sql))
                    print(f"Query executed successfully")
                except Exception as e:
                    print(e)
        elif str(row['Query']) != 'nan':
            df.loc[index, 'Query'] = format_query(row['Query'])
    return df
    
def find_sub_query(query):
    # Phân tích cú pháp SQL
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    parsed = sqlparse.parse(query)[0]

    # Tìm và thay thế WHERE NOT EXISTS bằng LEFT JOIN
    for i, token in enumerate(parsed.tokens):
        if token.ttype is None and isinstance(token, sqlparse.sql.Where):
            subquery = None
            # Tìm subquery bên trong WHERE NOT EXISTS
            for sub_token in token.tokens:
                if isinstance(sub_token, sqlparse.sql.Parenthesis):
                    subquery = sub_token
                    break
    return str(subquery)


def extract_conditions(query):
    # This pattern extracts conditions after 'WHERE' keyword or within the ON clause of a JOIN
    pattern = r'WHERE\s+(.*)|ON\s+(.*)'
    matches = re.findall(pattern, query, re.IGNORECASE)
    # Flattening the list of tuples and removing empty strings
    conditions = [condition for sublist in matches for condition in sublist if condition]
    return conditions
    
def extract_clean_columns(sql_query):
    # Extract the portion between SELECT and FROM using regex
    columns = re.findall(r'SELECT(.*?)FROM', sql_query, re.DOTALL)
    
    clean_columns = []
    column_ = []
    
    if columns:
        # Split the column string to get individual column names
        column_list = [col.strip() for col in columns[0].split(',')]
        
        for col in column_list:
            # Remove comments and extraneous whitespace
            clean_col = re.sub(r'--.*', '', col).strip()
            # Handle column aliases if 'AS' is used
            if 'AS' in clean_col:
                clean_col = clean_col.split('AS')[-1].strip()
            clean_columns.append(clean_col)           

    # Join the cleaned columns into a single string separated by commas
    # cleaned_column_string = ', '.join(clean_columns)
    
    for col in clean_columns:
        if ' ' in col:
            column_.append(col.split(' ')[-1])
        else:
            column_.append(col)
    
    return clean_columns, column_

def analyze_seq(df):
    # df['Expression'] = None
    # df['Properties'] = None
    index_to_drop = []
    for index, row in df.iterrows():

        if 'seq' in str(row['Query']):
            print(index)
            print("Sequence found")
            table_names = re.findall(r'FROM\s+([^\s]+)\s', row['Query'], re.IGNORECASE)

            variable_list = ['#CONNECTION_GSTX.GSTX_DBSCHEMA#', '#DYNAMICPARM.DATADT#', '#STATICPARM.SFDB2_DBSCHEMA#', '#STATICPARM.CURDTDB2_DBSCHEMA#', '#GSTX_LIMIT_CIF_OPEN_DT#']
            
            # nếu tìm thấy tên bảng có chứa variable_list thì thay thế bằng tên bảng 
            table_names = [reduce(lambda name, var: name.replace(var, '').replace('.', ''), variable_list, name) for name in table_names]
            
            cleaned_column_string, column_ = extract_clean_columns(row['Query'])
            
            sub_query = find_sub_query(row['Query']).replace('(', '').replace(')', '')
            
            raw_condition = extract_conditions(sub_query)
            
            rows=[]

            row_a = {
                "Job Name": row['Job Name'],
                "Stage Name": str(table_names[0]),
                "Stage Type ID": 'OracleConnectorPX',
                "Source Stage": '',
                "Target Stage": 'EXPRESSION'
            }
            rows.append(row_a)
            
            expression_expression = ', '.join([f"{table_names[0]}.{col}" if 'seq' not in col else col for col in cleaned_column_string])
            row_expression = {
                "Job Name": row['Job Name'],
                "Stage Name": 'EXPRESSION',
                "Stage Type ID": 'EXPRESSION',
                "Source Stage": str(table_names[0]),
                "Target Stage": 'JOIN',
                "Expression": expression_expression
            }
            rows.append(row_expression)
            
            row_b = {
                "Job Name": row['Job Name'],
                "Stage Name": str(table_names[1]),
                "Stage Type ID": 'OracleConnectorPX',
                "Source Stage": '',
                "Target Stage": 'JOIN'
            }
            rows.append(row_b)
            
            expression_join = ', '.join([f"EXPRESSION.{col}" for col in column_])
            condition = raw_condition[0].replace('a', 'EXPRESSION').replace('b', table_names[1])
            row_join_a = {
                "Job Name": row['Job Name'],
                "Stage Name": 'JOIN',
                "Stage Type ID": 'JOIN',
                "Source Stage": 'EXPRESSION',
                "Target Stage": 'FILTER',
                "Expression": str(expression_join),
                "Properties": f"""1 | LEFT_OUTER | {condition}"""
            }
            rows.append(row_join_a)
            
            row_join_b = {
                "Job Name": row['Job Name'],
                "Stage Name": 'JOIN',
                "Stage Type ID": 'JOIN',
                "Source Stage": str(table_names[1]),
                "Target Stage": 'FILTER',
                "Expression": str(expression_join),
                "Properties": f"""2 | LEFT_OUTER | {condition}"""
            }
            rows.append(row_join_b)
            
            # cắt ra tên 2 column name in condition
            column_name_in_filter = list(re.findall(r'\.\b(\w+)\b', condition))
            row_where = {
                "Job Name": row['Job Name'],
                "Stage Name": 'FILTER',
                "Stage Type ID": 'FILTER',
                "Source Stage": 'JOIN',
                "Target Stage": str(row['Target Stage']),
                "Properties": f"{table_names[1]}.{column_name_in_filter[0]} IS NULL",
            }
            
            rows.append(row_where)
            
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            
            
            # UPDATE df['Source Stage'] = index to 'FILTER'
            
            df.loc[df['Source Stage'] == row['Stage Name'], 'Source Stage'] = 'FILTER'
            
            
            index_to_drop.append(index)
    
    # Drop rows that contain sequences
    df = df.drop(index_to_drop)
            
    return df
    

        
def get_df_to_odi_input(df: pd.DataFrame) -> str:
    """
    Transforms a DataFrame to match the ODI input format and saves it to an Excel file.
    This function performs several transformations on the input DataFrame, including:
    - Correcting the query using the provided connection and rules.
    - Analyzing sequences.
    - Mapping job names to stage names based on specific conditions.
    - Updating the 'Physical' column based on stage type.
    - Converting certain columns to uppercase.
    - Renaming columns to match the ODI input format.
    - Replacing specific component type values with corresponding ODI values.
    - Sorting the DataFrame by 'Job Name' and 'Stage Name'.
    - Saving the transformed DataFrame to an Excel file.
    Args:
        df (pd.DataFrame): The input DataFrame containing job and stage information.
        connection (oracledb): The database connection object used for query correction.
        more_rule_db2_to_oracle (str): Additional rules for DB2 to Oracle conversion.
    Returns:
        str: The file path of the saved Excel file containing the transformed DataFrame.
    """
    # df = get_corrected_query(df, connection, more_rule_db2_to_oracle)
    
    df = analyze_seq(df)
    
    df['Physical'] = None
    
    # Iterate through each job name
    job_names = df['Job Name'].unique()

    # Get stage names with target = None or /20
    mapping = {}
    for job_name in job_names:
        df_job = df[(df['Job Name'] == job_name) & (df['Target Stage'].isnull() | df['Target Stage'].str.contains('/20'))]
        stage_names = df_job['Stage Name'].tolist()  # Get all matching stage names as a list
        
        if stage_names:  # If there are any stage names, take the first one
            mapping[job_name] = stage_names[0]
    
    for job_name, rows in df.iterrows():
        # Update the 'Physical' column for rows that meet the condition
        if rows['Stage Type ID'] == 'OracleConnectorPX' or rows['Stage Type ID'] == 'DB2ConnectorPX':
            df.loc[(df['Job Name'] == rows['Job Name']) & (df['Stage Name'] == rows['Stage Name']), 'Physical'] = mapping[rows['Job Name']]
        else:
            df.loc[(df['Job Name'] == rows['Job Name']) & (df['Stage Name'] == rows['Stage Name']), 'Physical'] = rows['Stage Name']
    
    df[['Physical', 'Stage Name', 'Job Name', 'Target Stage', 'Source Stage']] = df[['Physical', 'Stage Name', 'Job Name', 'Target Stage', 'Source Stage']].map(lambda x: x.upper() if isinstance(x, str) else x)
    
    df = df[['Job Name', 'Stage Name', 'Physical', 'Stage Type ID', 'Source Stage', 'Target Stage', 'Query', 'Query Usage', 'Expression', 'Properties']]
    
    df = df.sort_values(by=['Job Name', 'Stage Name', 'Stage Type ID'])
    
    # Rename columns: Job Name to Mapping Name, Stage Name to Component Name, Physical to Physical Name, Stage Type ID to Component Type ID, Source Stage to Source Component, Target Stage to Target Component, Query Usage to Query Usage
    df = df.rename(columns={'Job Name': 'Mapping Name', 'Stage Name': 'Component Name', 'Physical': 'Physical Name', 'Stage Type ID': 'Component Type', 'Source Stage': 'Source Component', 'Target Stage': 'Target Component'})
    
    # Replace values in the Component Type column with corresponding values
    # OracleConnectorPX -> DATASTORE, DB2ConnectorPX -> DATASTORE, PxFunnel -> SET
    df['Component Type'] = df['Component Type'].replace({'OracleConnectorPX': 'DATASTORE', 'DB2ConnectorPX': 'DATASTORE', 'PxFunnel': 'SET'})
    
    
    file_path_odi_input = r'./extract_datastage/odi_input.xlsx'
    
    df.to_excel(file_path_odi_input, index=False)
    
    return file_path_odi_input


def main(): 
    
    # Create a result file
    file_result_path = create_result_file(r'./extract_datastage/ANALYZE/SUMMARY')
    
    # Read the result file
    df = create_dataframe(file_result_path)
    
    # more_rule_db2_to_oracle = read_file(r'./extract_datastage/more_rules/more_rule_db2_to_oracle.txt')
    
    file_path = get_df_to_odi_input(df)
    
    print(file_path)
    
    return file_path


if __name__ == '__main__':
    main()
