import os
import time
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from prophet import Prophet
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from generic import all_plant_dc_combos, format_script_runtime, create_output_folders


def reg_model(sku,
                plant,
                dc,
                forecast_from_date,
                forecast_till_date,
                df_plant_level,
                df_amount_agg,
                save_missing_dates_plots = False,
                save_performance_plots = False,
                ):
    
    forecast_from_date = pd.to_datetime(forecast_from_date)
    forecast_till_date = pd.to_datetime(forecast_till_date)
    # total amount subset of the SKU combo
    df_sub_amt = df_amount_agg[(df_amount_agg['plant'] == plant)
                                & (df_amount_agg['distribution_channel'] == dc)
                                & (df_amount_agg['parent_name'] == sku)]
    # order quantity subset of the SKU combo
    df_sub_qty = df_plant_level[(df_plant_level['plant'] == plant)
                                & (df_plant_level['distribution_channel'] == dc)
                                & (df_plant_level['parent_name'] == sku)]
    SKU_CODE = df_sub_qty['parent_code'].unique()[0]
    # Output folder path
    outputs_sub_folder_path = create_output_folders("model1", forecast_from_date, forecast_till_date)
    
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
    train = df_sub_qty[df_sub_qty['delivery_date'] < forecast_from_date][['delivery_date', "quantity"]].rename(columns = {"delivery_date":"ds", "quantity":"y"})
    test = df_sub_qty[df_sub_qty['delivery_date'] >= forecast_from_date][['delivery_date', "quantity"]].rename(columns = {"delivery_date":"ds", "quantity":"y"})
    # First lag of total_amount to pass as regressor
    df_sub_amt.loc[:,'total_amount'] = df_sub_amt['total_amount'].shift(1, fill_value = 0)
    reg = df_sub_amt[['delivery_date', "total_amount"]].rename(columns = {"delivery_date":"ds", "total_amount":"x"})
    if reg['ds'].max() < forecast_till_date:
        historical_avg_amount = reg['x'].mean()
        historical_std_amount = reg['x'].std()
        future_reg_values = pd.DataFrame(pd.date_range(reg['ds'].max(),
                                                        forecast_till_date,
                                                        name = "ds"))
        future_reg_values['x'] = np.random.normal(historical_avg_amount, historical_std_amount, future_reg_values.shape[0])
        reg = pd.concat([reg, future_reg_values],
                        axis = "index",
                        join = "inner",
                        ignore_index = True)
        reg['x'] = reg.mask(reg['x']<0, historical_avg_amount)['x'].values
    # if only one datapoint then foreacst that point throughout the forecast horizon
    if train.shape[0] == 1:
        forecasts = pd.DataFrame(pd.date_range(forecast_from_date,
                                                forecast_till_date,
                                                name = "ds")) 
        forecasts['yhat'] = train['y'].mean()
    # If no data exits in the train dataframe then forecast 1
    elif train.shape[0] == 0:
        forecasts = pd.DataFrame(pd.date_range(forecast_from_date,
                                                forecast_till_date,
                                                name = "ds")) 
        forecasts['yhat'] = 1
    # Else run the Prophet model
    else:
        model = Prophet()
        train = train.merge(reg, on = "ds")
        model.add_regressor("x")
        model.fit(train)
        future = model.make_future_dataframe(periods = len(pd.date_range(train['ds'].max(),
                                                                        forecast_till_date)))
        future = future.merge(reg, on = "ds")
        forecasts = model.predict(future) 
        replace_neg_with_this = train['y'].median()
        forecasts['yhat'] = forecasts.mask(forecasts['yhat'] < 0, replace_neg_with_this)['yhat'].values

    # OUTPUTS
    # saving performance plots
    if save_performance_plots:
        plt.clf()
        if test.shape[0] != 0:
            test_and_forecast = test.merge(forecasts[['ds','yhat']],
                                            on = "ds",
                                            how = "inner")
            MAPE = np.mean(abs((test_and_forecast['y'].values - test_and_forecast['yhat'].values)/test_and_forecast['y'].values))*100
        else:
            MAPE = "Test data not available"
        plt.plot(train['ds'], train['y'])
        if test.shape[0] != 0:
            plt.plot(test['ds'], test['y'])
            plt.plot(test_and_forecast['ds'], test_and_forecast['yhat'])
            plt.legend(['Train', "Test", 'Forcasts'], fontsize = 17)
        else:
            plt.plot(forecasts[forecasts['ds']>=forecast_from_date]['ds'], forecasts[forecasts['ds']>=forecast_from_date]['yhat'])
            plt.legend(['Train', "Forecasts"], fontsize = 17)
        if MAPE != "Test data not available":
            plt.title(f"{sku}\nLag = {1}\nMAPE = {round(MAPE,3)}", fontsize = 19)
        else:
            plt.title(f"{sku}\nLag = {1}\nMAPE = {MAPE}", fontsize = 19)
        plt.grid("on")
        plt.axhline(y=0, color = "black")
        os.makedirs(os.path.join(outputs_sub_folder_path, f"model_performance_plots_{plant}_{dc}"), exist_ok = True)
        plt.savefig(os.path.join(outputs_sub_folder_path, f"model_performance_plots_{plant}_{dc}",sku), dpi = 100, transparent = False)
        
    # SKU combo forecast output
    dates_df = pd.DataFrame(pd.date_range(forecast_from_date, forecast_till_date, name = "ds"))  
    Date_and_Quantity = forecasts.merge(dates_df, on = "ds", how = "inner")
    output = pd.DataFrame({"ItemCode":SKU_CODE,
                            "Material":sku,
                            "Plant":plant,
                            "Distribution_channel":dc,
                            "Date":Date_and_Quantity['ds'],
                            "Quantity":Date_and_Quantity['yhat']
                            })
    
    os.makedirs(os.path.join(outputs_sub_folder_path, f"model_outputs_{plant}_{dc}"), exist_ok = True)
    output.to_csv(os.path.join(outputs_sub_folder_path, f"model_outputs_{plant}_{dc}", f"{sku}_forecasts.csv".replace("/","-").replace("\\", "-")), index = False)
    
    return output


def model1(df_plant_level,
            df_amount_agg,
            forecast_from_date,
            forecast_till_date,
            data_plant_names,
            data_mapping,
            save_missing_dates_plots = False,
            save_performance_plots = False
            ):
    """
    Runs model1 for all combos
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
                out = reg_model(sku = SKU,
                                plant = PLANT,
                                dc = DC,
                                forecast_from_date = forecast_from_date,
                                forecast_till_date = forecast_till_date,
                                df_plant_level = df_plant_level,
                                df_amount_agg = df_amount_agg,
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
            outputs_sub_folder_path = create_output_folders("model1", pd.to_datetime(forecast_from_date), pd.to_datetime(forecast_till_date)) 
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

