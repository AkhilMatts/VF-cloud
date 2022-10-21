import os
import configparser
import datetime
import warnings
warnings.filterwarnings("ignore")

from pymongo import MongoClient
import pandas as pd

from model1 import model1
from model2 import model2
from generic import preprocess_SO_data, post_process, create_output_folders

def get_collection(uri, db, coll_name):
    client = MongoClient(host = uri)
    collection = client[db][coll_name]
    return collection

def fetch_and_save_SO_data(conn_str_so, db_so, collection_name_so):
    data_so =  get_collection(conn_str_so, db_so, collection_name_so)
    data_so = data_so.find({"req_del_date":{"$gte":"2021-09-01"},
                            "sales_document_type":{"$eq":"ZDOM"},
                            "sales_organization":{"$eq":5000},
                            "distribution_channel":{"$in":[50,51,52]},
                            "plant":{"$in":[1000,1009,1028,1020,1056,1017,1049,1031]},
                            # "cancel_status":{"$ne":2}
                            })
    df = pd.json_normalize(list(data_so),
                            record_path = "item",
                            meta = ['req_del_date', "created_at", "order_date",
                                    "sold_to_party_description", "sold_to_party", "sales_order_no",
                                    "distribution_channel"])
    df = df[df['rejection_reason'] !='W1']
    df = df.drop_duplicates(subset=['material_no','sales_order_no','order_quantity'],keep='last')
    if os.path.exists(os.path.join(os.getcwd(), "df_all_plants.csv")):
        os.remove(os.path.join(os.getcwd(), "df_all_plants.csv")) 
    df.to_csv(os.path.join(os.getcwd(), "df_all_plants.csv"),
                index = False)

def fetch_and_save_CP_data(conn_str_cp, db_cp, collection_name_cp):
    data_cp = get_collection(conn_str_cp, db_cp, collection_name_cp)
    data_cp = data_cp.find()
    cp_df = pd.json_normalize(list(data_cp))
    cp_df = cp_df.drop_duplicates()
    if os.path.exists(os.path.join(os.getcwd(), "child_parent_mapping.csv")):
            os.remove(os.path.join(os.getcwd(), "child_parent_mapping.csv")) 
    cp_df.to_csv(os.path.join(os.getcwd(), "child_parent_mapping.csv"),
            index = False)

def update_config_file_dates(config_obj, 
                            train_till_date,
                            forecast_from_date, 
                            forecast_till_date,
                            file_name = "config.ini"):
    """
    All dates to be in YYYY-MM-DD format
    """
    config_obj.set("DATE_PARAMETERS", "train_till_date", train_till_date)
    config_obj.set("DATE_PARAMETERS", "forecast_from_date", forecast_from_date)
    config_obj.set("DATE_PARAMETERS", "forecast_till_date", forecast_till_date)
    config_file_path = os.path.join(os.getcwd(), file_name)
    with open(config_file_path, "w") as file:
        config_obj.write(file)

def run():
    config = configparser.ConfigParser()
    try: 
        print("Reading config file...")
        out = config.read(os.path.join(os.getcwd(), "config.ini"))
        date_time_format = r"%Y-%m-%d"
        old_train_till_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['train_till_date'], date_time_format).date()
        old_forecast_from_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['forecast_from_date'], date_time_format).date()
        old_forecast_till_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['forecast_till_date'], date_time_format).date()
    except KeyError:
        if len(out) == 0:
            return "'config.ini' not found in working directory"
        else:
            return "Parameter not found in config file"
    except ValueError:
        return "Datetime format is wrong in the config file"
    
    CURRENT_DATE = datetime.datetime.today().date()
    if (old_forecast_from_date < CURRENT_DATE) and (CURRENT_DATE < old_forecast_till_date):
        # Update the config file's date parameters
        new_train_till_date = CURRENT_DATE
        new_forecast_from_date = old_forecast_till_date + datetime.timedelta(days = 1)
        new_forecast_till_date = new_forecast_from_date + datetime.timedelta(days = 6)
        update_config_file_dates(config, str(new_train_till_date), str(new_forecast_from_date), str(new_forecast_till_date))
        print("Read and updated successfully")

        # Read the updated dates
        config = configparser.ConfigParser()
        out = config.read(os.path.join(os.getcwd(), "config.ini"))
        date_time_format = r"%Y-%m-%d"
        train_till_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['train_till_date'], date_time_format).date()
        forecast_from_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['forecast_from_date'], date_time_format).date()
        forecast_till_date = datetime.datetime.strptime(config['DATE_PARAMETERS']['forecast_till_date'], date_time_format).date()
        
        # FETCHING DATA FROM MONGODB 
        # Fetch most recent S.O. data
        print("Fetching SO data...")
        CONN_STR_SO = config['SO_DATA']['conn_str_so']
        DB_SO = config['SO_DATA']['db_so']
        COLLECTION_NAME_SO = config['SO_DATA']['collection_name_so']
        fetch_and_save_SO_data(CONN_STR_SO, DB_SO, COLLECTION_NAME_SO)
        # Fetch most recent Child parent mapping 
        print("Fetching CP data...")
        CONN_STR_CP = config['CP_DATA']['conn_str_cp']
        DB_CP = config['CP_DATA']['db_cp']
        COLLECTION_NAME_CP = config['CP_DATA']['collection_name_cp']
        fetch_and_save_CP_data(CONN_STR_CP, DB_CP, COLLECTION_NAME_CP)

        # READING DATA 
        # Read the S.O data
        print("Reading Data...")
        so_data = pd.read_csv("df_all_plants.csv")
        # Read the child parent mapping data
        data_mapping = pd.read_csv("child_parent_mapping.csv")
        # Read the plant with names data
        data_plant_names =  pd.read_csv("plant_with_name.csv")
        # Read the required SKUs data
        data_sku = pd.read_csv("required_sku.csv")
        
        # PREPROCESSING
        print("Preprocessing...")
        df_plant_level = preprocess_SO_data(so_data, data_mapping, data_plant_names, data_sku, aggregate_over = "quantity", only_required_skus = False)
        df_amount_agg = preprocess_SO_data(so_data, data_mapping, data_plant_names, data_sku, aggregate_over = "total_amount", only_required_skus = False)
        
        # RUNNING THE MODELS
        # run model 1
        print("Running model 1...")
        forecasts_1, time_taken_1 = model1(df_plant_level, df_amount_agg, forecast_from_date, forecast_till_date, data_plant_names, data_mapping, save_missing_dates_plots = False, save_performance_plots = False)
        output_model1 = post_process(forecasts_1, time_taken_1, str(train_till_date))
        outputs_sub_folder_path = create_output_folders("model1", forecast_from_date, forecast_till_date)
        output_model1.to_csv(os.path.join(outputs_sub_folder_path, "output_akhil.csv"), index = False)        
        # run model 2
        print("Running model 2...")
        forecasts_2, time_taken_2 = model2(df_plant_level,train_till_date, forecast_from_date, forecast_till_date, data_plant_names, data_mapping, save_missing_dates_plots = False, save_performance_plots = False)
        output_model2 = post_process(forecasts_2, time_taken_2, str(train_till_date))
        outputs_sub_folder_path = create_output_folders("model2", forecast_from_date, forecast_till_date)
        output_model2.to_csv(os.path.join(outputs_sub_folder_path,"output_akhil_2.csv"), index = False)
        print("Done!")

        # INSERT INTO COLLECTION 
        # model 1 
        OUTPUT_CONN_STR = config['OUTPUT']['conn_str']
        OUTPUT_DB = config['OUTPUT']['db']
        MODEL1_COLLECTION_NAME = config['OUTPUT']['model1_collection_name']
        output = MongoClient(host = OUTPUT_CONN_STR)
        output[OUTPUT_DB][MODEL1_COLLECTION_NAME].insert_many(output_model1.to_dict("records"))
        # model 2
        MODEL2_COLLECTION_NAME = config['OUTPUT']['model2_collection_name']
        output[OUTPUT_DB][MODEL2_COLLECTION_NAME].insert_many(output_model2.to_dict("records"))
    else:
        return "Can't run the script"

if __name__ == "__main__":
    run()