from collections import defaultdict
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


def preprocess_SO_data(so_data,
                        data_mapping,
                        data_plant_names,
                        data_sku,
                        aggregate_over = "quantity",
                        only_required_skus = True):
    """
    Sales order preprocessing pipeline
    
    Parameters:
    -----------
    so_data: DataFrame. Any raw sales order data from mongoDB. Usually, SAP sales orders or MDB sales orders
    data_mapping: DataFrame. Child parent mapping
    data_plant_names: DataFrame. Plant and their respective names
    data_sku: DataFrame. Required SKUs 
    aggregate_over: str. The variable to aggregate (at PLANT-DC-SKU-DATE level) over. One of ["quantity", "sales_order_no", "total_amount"]
    only_required_skus: bool. If to only use required skus

    Returns:
    --------
    Dataframe. Per day aggregated values of order quantity, total_amount, or count of sales orders
    """
    plant_level = ["plant", "distribution_channel"]
    material_level = ["material_no", "material_description", "uom"]
    customer_level = ["req_del_date","order_quantity", "total_amount", 
                        "sold_to_party_description", "sold_to_party","sales_order_no", 
                        "order_date"]

    so_data = so_data[plant_level + material_level + customer_level]
    so_data = so_data.rename(columns={"req_del_date": "delivery_date"})

    # Getting all child SKUs
    material_list = list(data_mapping['material_number'].unique())
    so_data = so_data[so_data['material_no'].isin(material_list)] 
    # Removing scrapped and null materials 
    null_ind = so_data[so_data['material_description'].isnull()].index
    so_data.loc[null_ind, 'material_description'] = ''
    so_data['material_description'] = so_data['material_description'].apply(lambda x : 'remove' if 'scrapped' in x.lower().split() else x)
    so_data = so_data[so_data['material_description'] != 'remove']
    # adding plant names
    data_plant_names = data_plant_names[['plant_code', 'plant_name']]
    data_plant_names.loc[:,'plant_code'] = data_plant_names['plant_code'].astype(int)
    so_data = pd.merge(so_data, data_plant_names, how = 'left', left_on = 'plant', right_on = 'plant_code')
    # child parent mapping
    so_data_merge = pd.merge(so_data, data_mapping, how = 'left', left_on=['material_no','uom'], right_on = ['material_number','uom'])
    # converting order quantity
    so_data_merge.loc[:,'net_weight'] = so_data_merge['net_weight'].astype(float)
    so_data_merge['order_quantity'].replace('', 0.0, inplace = True)
    so_data_merge.loc[:,'order_quantity'] = so_data_merge['order_quantity'].astype(float)
    so_data_merge['quantity'] = so_data_merge['order_quantity'] * so_data_merge['net_weight']
    so_data_merge['total_amount'].replace('',0,inplace=True)

    # AGGREGATIONS
    def aggregate(so_data_merge, agg_var, data_sku, agg_func, only_required_skus = True):
        so_data_merge_grp = so_data_merge.groupby(['plant', 'plant_name', 'parent_code','parent_name',
                                                'parent_uom','delivery_date','distribution_channel']).agg({agg_var:agg_func}).reset_index()
        # selecting required skus and removing sku as per business
        remove = np.array(['WC0000000000200011','WC0000000000200034','WC0000000000200035','WC0000000000200024',
                        'WC0000000000260003','WC0000000000260016','WC0000000000250001','WC0000000000240002'])
        if only_required_skus:
            sku_list = data_sku['material_code'].to_list()
            so_data_merge_grp = so_data_merge_grp[so_data_merge_grp['parent_code'].isin(sku_list)]
        df_agg = so_data_merge_grp[~so_data_merge_grp['parent_code'].isin(remove)]
        df_agg.loc[:, agg_var] = df_agg[agg_var].astype(float)
        df_agg['delivery_date'] = pd.to_datetime(df_agg['delivery_date'])
        return df_agg

    if aggregate_over == "quantity":
        df_plant_level = aggregate(so_data_merge = so_data_merge, 
                                agg_var = aggregate_over, 
                                data_sku = data_sku, 
                                agg_func = "sum",
                                only_required_skus = only_required_skus)
        return df_plant_level

    elif aggregate_over == "sales_order_no":
        df_so_no_agg = aggregate(so_data_merge = so_data_merge,
                                agg_var = aggregate_over, 
                                data_sku = data_sku,
                                agg_func = "count",
                                only_required_skus = only_required_skus)
        return df_so_no_agg

    elif aggregate_over == "total_amount":
        df_amount_agg = aggregate(so_data_merge = so_data_merge,
                                agg_var = aggregate_over, 
                                data_sku = data_sku,
                                agg_func = "sum",
                                only_required_skus = only_required_skus)
        return df_amount_agg


def all_plant_dc_combos(df_plant_level):
    """
    Get all plant-dc combinations as a dict. Sales order data to be passed as argument 
    """
    dcs_in_plants = (df_plant_level
                        .groupby(["plant", "distribution_channel"])[["quantity"]]
                        .agg("count")
                        .drop("quantity", axis = "columns")
                        )
    d = defaultdict(list)
    for plant, dc in list(dcs_in_plants.index):
        d[plant].append(dc)
    return d

def format_script_runtime(x):
    """
    Converts seconds into 'hh:mm'
    """
    hours = str(int(x//3600))
    minutes = str(round(x%3600/60,0))[:-2]
    if len(hours) == 1:
        if len(minutes) == 2:
            return f"0{hours}:{minutes}"
        elif len(minutes) == 1:
            return f"0{hours}:0{minutes}"
    else:
        if len(minutes) == 2:
            return f"{hours}:{minutes}"
        else:
            return f"{hours}:0{minutes}"


# Output folder path
def create_output_folders(model_name, forecast_from_date, forecast_till_date):
    """
    Parameters:
    -----------
    model_name: str. ex:model1, model2
    forecast_from_date: datetime. 
    forecast_till_date: datetime.

    Returns:
    --------
    subfolder path
    """
    outputs_sub_folder_path = os.path.join(os.getcwd(),
                                            f"outputs_{model_name}",
                                            forecast_from_date.strftime("%b %Y"),
                                            (forecast_from_date.strftime("%b %d")
                                            + '-'
                                            + forecast_till_date.strftime("%b %d")
                                            ))
    os.makedirs(outputs_sub_folder_path, exist_ok = True)
    return outputs_sub_folder_path


def post_process(full_data,
                    time_taken_for_all_SKUs,
                    last_train_date):
    """
    Post-process the data to insert into the collection

    Parameters:
    -----------
    full_data: dataframe. output from all the combos
    time_taken_for_all_SKUs: int. Time in seconds
    last_train_date: str. In YYYY-MM-DD format

    Returns:
    --------

    """
    full_data = full_data.rename(columns = {"Distribution_channel":"distribution_channel",
                                            "ItemCode":"material_number",
                                            "Material":"material_name",
                                            "parent_uom":"uom",
                                            "Date":"date",
                                            "Quantity":"forecast_quantity"})
    full_data["uid"] = full_data['plant_code'].astype(str) + "_" + full_data['material_number']
    full_data["uuid"] = full_data['plant_code'].astype(str) + "_" + full_data['material_number'] + '_' + full_data['distribution_channel'].astype(str)
    full_data['run_date'] = pd.to_datetime(datetime.date.today())
    full_data['script_runtime'] = format_script_runtime(time_taken_for_all_SKUs)
    full_data = full_data[["plant_code", "plant_name", "distribution_channel",
                        "material_number", "material_name", "uom", 
                        "date", "forecast_quantity", "uid", "uuid",
                        "run_date", "script_runtime"]]
    full_data = full_data.sort_values(['plant_code', "distribution_channel", "material_name", "date"])
    full_data = full_data.reset_index(drop = True)
    full_data['last_train_date'] = pd.to_datetime(last_train_date)

    # Removing duplicates:
    full_data = full_data.drop_duplicates(subset = ['uuid', 'date'], keep = "first")
    return full_data
