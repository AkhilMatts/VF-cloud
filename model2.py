import os
import time
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from xgboost import XGBRegressor

from generic import all_plant_dc_combos, format_script_runtime, create_output_folders


def generate_features(df,
                        date_col = "delivery_date"):
    """
    Generates date features

    Parameters:
    -----------
    df: dataframe. Must have a datetime column
    """
    x = df.copy()
    x = x.sort_values(date_col)
    x["month"] = x[date_col].dt.month
    x['week_num'] = x[date_col].dt.week
    x['year'] = x[date_col].dt.year
    x["day"] = x[date_col].dt.day
    x['day_of_week'] = x[date_col].dt.weekday
    x['weekend_dummy'] =  (x[date_col].dt.day_name()
                                        .isin(['Sunday', "Saturday"])
                                        .replace({True:1,
                                                False:0})).astype(int)
    x = x.reset_index(drop = True)
    return x


def xgb_model(sku,
                plant,
                dc,
                train_till_date,
                forecast_from_date,
                forecast_till_date,
                df_plant_level,
                save_missing_dates_plots = False,
                save_performance_plots = False,
                ):
    
    train_till_date = pd.to_datetime(train_till_date)
    forecast_from_date = pd.to_datetime(forecast_from_date)
    forecast_till_date = pd.to_datetime(forecast_till_date)
    # order quantity subset of the SKU combo
    df_sub_qty = df_plant_level[(df_plant_level['plant'] == plant)
                                & (df_plant_level['distribution_channel'] == dc)
                                & (df_plant_level['parent_name'] == sku)]
    SKU_CODE = df_sub_qty['parent_code'].unique()[0]
    # Output folder path
    outputs_sub_folder_path = create_output_folders("model2", forecast_from_date, forecast_till_date)
    # saving missing dates plots
    if save_missing_dates_plots:
        plt.clf()
        reindexed_series = df_sub_qty.set_index('delivery_date').reindex(pd.date_range(df_sub_qty['delivery_date'].min(),
                                                                                        df_sub_qty['delivery_date'].max()))['quantity']
        num_missing_dates = reindexed_series.isnull().sum()
        percentage = round(100*num_missing_dates/reindexed_series.shape[0], 2)
        
        fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True)
        fig.suptitle(sku + f"\nCode = {SKU_CODE}, Plant = {plant}, DC = {dc} (Aggregated series)", fontsize = 19)
        ax[0].plot(df_sub_qty['delivery_date'], df_sub_qty['quantity'])
        reindexed_series.plot.line(ax = ax[1])
        ax[1].set_title(f"{num_missing_dates} ({percentage}%) Missing dates", fontsize = 17)
        os.makedirs(os.path.join(outputs_sub_folder_path, f"SKU_and_missing_dates_{plant}_{dc}"), exist_ok = True)
        plt.savefig(os.path.join(outputs_sub_folder_path, f"SKU_and_missing_dates_{plant}_{dc}", sku), dpi = 100, transparent = False)

    # MODEL
    # Train-test split
    train = df_sub_qty[df_sub_qty['delivery_date'] <= train_till_date]
    train = generate_features(train).drop(['plant', 'plant_name', 'parent_code','distribution_channel',
                                            'parent_uom', 'delivery_date',  'parent_name'] ,axis = "columns")
    test = df_sub_qty[df_sub_qty['delivery_date'].between(forecast_from_date, forecast_till_date)]


    if (test['delivery_date'].max() < forecast_from_date) or (test.shape[0] < len(pd.date_range(forecast_from_date, forecast_till_date))):
        test = (test
                .set_index('delivery_date')
                .reindex(pd.date_range(forecast_from_date, forecast_till_date))
                .reset_index()
                .rename(columns = {"index":"delivery_date"}))


    test = generate_features(test).drop(['plant', 'plant_name', 'parent_code','distribution_channel',
                                            'parent_uom', 'delivery_date', 'parent_name'] ,axis = "columns")

    X_train = train.drop("quantity", axis = "columns")
    Y_train = train['quantity']
    X_test = test.drop('quantity', axis = "columns")
    Y_test = test['quantity']
    
    if not X_train.shape[0] <= 1:
        xg = XGBRegressor(learning_rate = 0.01,
                            max_depth = 8,
                            min_child_weight = 7,
                            gamma = 0.0,
                            colsample_bytree = 0.4)
        xg.fit(X_train, Y_train)
        forecasts = xg.predict(X_test)
        output = pd.DataFrame(pd.date_range(forecast_from_date, forecast_till_date, name = "Date"))
        output['Quantity'] = forecasts
        output['ItemCode'] = SKU_CODE
        output['Material'] = sku
        output['Plant'] = plant
        output['Distribution_channel'] = dc
        # Replacing negative forecasts with historical_avg_amount
        historical_avg_amount = Y_train.mean()
        output['Quantity'] = output.mask(output['Quantity']<0, historical_avg_amount)['Quantity'].values
    else:
        output = pd.DataFrame(pd.date_range(forecast_from_date, forecast_till_date, name = "Date"))
        output['Quantity'] = 1
        output['ItemCode'] = SKU_CODE
        output['Material'] = sku
        output['Plant'] = plant
        output['Distribution_channel'] = dc


    # # OUTPUTS
    # # saving performance plots
    # if save_performance_plots:
    #     plt.clf()
    #     if test.shape[0] != 0:
    #         test_and_forecast = test.merge(forecasts[['ds','yhat']],
    #                                         on = "ds",
    #                                         how = "inner")
    #         MAPE = np.mean(abs((test_and_forecast['y'].values - test_and_forecast['yhat'].values)/test_and_forecast['y'].values))*100
    #     else:
    #         MAPE = "Test data not available"
    #     plt.plot(train['ds'], train['y'])
    #     if test.shape[0] != 0:
    #         plt.plot(test['ds'], test['y'])
    #         plt.plot(test_and_forecast['ds'], test_and_forecast['yhat'])
    #         plt.legend(['Train', "Test", 'Forcasts'], fontsize = 17)
    #     else:
    #         plt.plot(forecasts[forecasts['ds']>=forecast_from_date]['ds'], forecasts[forecasts['ds']>=forecast_from_date]['yhat'])
    #         plt.legend(['Train', "Forecasts"], fontsize = 17)
    #     if MAPE != "Test data not available":
    #         plt.title(f"{SKU}\nLag = {1}\nMAPE = {round(MAPE,3)}", fontsize = 19)
    #     else:
    #         plt.title(f"{SKU}\nLag = {1}\nMAPE = {MAPE}", fontsize = 19)
    #     plt.grid("on")
    #     plt.axhline(y=0, color = "black")
    #     os.makedirs(os.path.join(outputs_sub_folder_path, f"model_performance_plots_{plant}_{dc}"), exist_ok = True)
    #     plt.savefig(os.path.join(outputs_sub_folder_path, f"model_performance_plots_{plant}_{dc}",sku), dpi = 100, transparent = False)
        
    # SKU combo forecast output 
    os.makedirs(os.path.join(outputs_sub_folder_path, f"model_outputs_{plant}_{dc}"), exist_ok = True)
    output.to_csv(os.path.join(outputs_sub_folder_path, f"model_outputs_{plant}_{dc}", f"{sku}_forecasts.csv".replace("/","-").replace("\\", "-")), index = False)
    
    return output

def model2(df_plant_level,
            train_till_date,
            forecast_from_date,
            forecast_till_date,
            data_plant_names,
            data_mapping,
            save_missing_dates_plots = False,
            save_performance_plots = False
            ):
    """
    Runs model2 for all combos
    """
    PLANTs_and_DCs = all_plant_dc_combos(df_plant_level)
    start_whole = time.perf_counter()
    for PLANT in tqdm.tqdm(PLANTs_and_DCs.keys(), "PLANTs completed"):
        DCs = PLANTs_and_DCs[PLANT]
        for DC in tqdm.tqdm(DCs, "DCs completed"): 
            unique_SKUs_in_PLANT_DC_combo = df_plant_level[(df_plant_level['plant'] == PLANT)
                                                            & (df_plant_level['distribution_channel'] == DC)]['parent_name'].unique()
            final = pd.DataFrame()
            for SKU in tqdm.tqdm(unique_SKUs_in_PLANT_DC_combo, "SKUs COMPLETED"):
                out = xgb_model(sku = SKU,
                                plant = PLANT,
                                dc = DC,
                                train_till_date = train_till_date,
                                forecast_from_date = forecast_from_date,
                                forecast_till_date = forecast_till_date,
                                df_plant_level = df_plant_level,
                                save_missing_dates_plots = save_missing_dates_plots,
                                save_performance_plots = save_performance_plots,
                                )
                if final.shape[0] == 0:
                    final = out
                else:
                    final = pd.concat([final, out],
                                        axis = "index",
                                        join = "inner",
                                        ignore_index = True)
            final = final.sort_values(["Material","Date"]) 
            outputs_sub_folder_path = create_output_folders("model2", pd.to_datetime(forecast_from_date), pd.to_datetime(forecast_till_date))
            os.makedirs(os.path.join(outputs_sub_folder_path, "final_outputs"), exist_ok = True)
            final.to_csv(os.path.join(outputs_sub_folder_path, "final_outputs", f"model_results_{PLANT}_{DC}.csv"), index = False)
    end_whole = time.perf_counter()
    time_taken_for_all_SKUs = end_whole - start_whole
    in_minutes = time_taken_for_all_SKUs / 60
    print(f"Time taken to run the model for all SKUs is: \
        \n{round(in_minutes, 5)} mins\
        \nor\nFormatted (hh:mm) = {format_script_runtime(time_taken_for_all_SKUs)}")

    # read all the csv files and concatinate them
    for idx, path in enumerate(glob.glob(os.path.join(outputs_sub_folder_path, "final_outputs", "*.csv"))):
        df = pd.read_csv(path)
        if idx == 0:
            full_data = df
        else:
            full_data = pd.concat([df, full_data],
                                    axis = "index",
                                    ignore_index = True,
                                    join = "inner")
    full_data['ItemCode'] = full_data['ItemCode'].astype(str)
    full_data['Plant'] = full_data['Plant'].astype(np.int64)
    full_data['Distribution_channel'] = full_data['Distribution_channel'].astype(int)
    full_data['Date'] = pd.to_datetime(full_data['Date'])
    full_data = full_data.merge(data_plant_names[['plant_code', 'plant_name']],
                                left_on = "Plant",
                                right_on = "plant_code",
                                how = "inner")
    full_data = full_data.merge(data_mapping[['parent_code', 'parent_uom']].drop_duplicates(),
                                how = "inner",
                                left_on = "ItemCode",
                                right_on = "parent_code")
    full_data = full_data.drop("Plant", axis = "columns")
    return full_data, time_taken_for_all_SKUs

