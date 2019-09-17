
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



def load_data():
    pd.set_option('display.expand_frame_repr', False)

    data = pd.read_csv('nyc-rolling-sales.csv', index_col=0)
    # rename columns
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    print(data.head())
    return data


def analyze_data(data):
    print()
    print("Number of rows: ", data.shape[0])
    print("Number of columns: ", data.shape[1])
    print("We could like to know data type of each column, minx/max values of columns, format of columns, interested columns")
    print("We see that....")
    print("0 in land square feet, gross square feet, year built....")
    print("Too old year built, such as 1111, 1680, 1800")

    print("Do we care address, apartment number?")

    return



def cleanData(data):
    print()
    print("##### clean up data")
    print("Format data, such as replace - with null, convert string to number (since the algorithm takes number as input)")

    data['price'] = data.sale_price.map(lambda x: sale(x))
    data['sale_date'] = pd.to_datetime(data['sale_date'])
    data['land_square'] = data.land_square_feet.map(lambda x: sale(x))
    data['gross_square'] = data.gross_square_feet.map(lambda x: sale(x))
    data['address'] = data.address.map(lambda x: address(x))

    


    data.sale_date = [item.to_julian_date() for item in data.sale_date]

    # print(data['building_class_category'].value_counts().head())
    # print(len(data))


    data.building_class_category = data.building_class_category.str.strip().str.lower().str.replace(' ', '').str.replace('-', '_')
    data = data.drop_duplicates(data.columns, keep='last')

    data.neighborhood = data.neighborhood.map(lambda x: convertToInt(x, neighborhood_map))
    data.building_class_category = data.building_class_category.map(
        lambda x: convertToInt(x, building_class_map))

    data.tax_class_at_present = data.tax_class_at_present.map(
        lambda x: convertToInt(x, tax_class_at_present_map))
    data.building_class_at_present = data.building_class_at_present.map(
        lambda x: convertToInt(x, building_class_at_present_map))
    data.building_class_at_time_of_sale = data.building_class_at_time_of_sale.map(
        lambda x: convertToInt(x, building_class_at_time_of_sale_map))

    return data


def data_engineer(data):
    print()
    print("We would like to derive new features from existing features, such as extract year, month from sale_date.")
    print("We would like to append external data, such as append APR, GPD, interests, anything that might have impact on house prices.")


def build_model(data):
    print()
    print("Columns: "+data.columns)
    print(data.head())
    features = ['land_square', 'gross_square', 'year_built',
                'district',
                'neighborhood',
                'tax_class_at_present',
                'building_class_category', 'building_class_at_present',
                'building_class_at_time_of_sale',
                'residential_units', 'commercial_units', 'total_units', 'price']



    data['price'] = np.log1p(data['price'])
    data['land_square'] = data['land_square']
    data['gross_square'] = data['gross_square']

    X = data[features]
    print(X.head())
    # X.to_csv('cleaned_house_data.csv')

    X = X[X.gross_square != 0]

    X = X[X.land_square != 0]
    X = X[X.price > np.log1p(100000)]

    X = X[X.year_built > 1880]

    X = X[X.building_class_category >= 0]

    X = X[X.district >= 0]
    X = X[X.neighborhood >= 0]
    X = X[X.tax_class_at_present >= 0]
    X = X[X.building_class_at_present >= 0]
    X = X[X.building_class_at_time_of_sale >= 0]

    y = X['price']
    X2 = X.drop(['price'], axis=1).values

    train_X, val_X, train_y, val_y = train_test_split(X2, y, test_size=0.25)

    model = RandomForestRegressor()
    model.fit(train_X, train_y)

    y_pred = model.predict(val_X)

    # print("Feature Importances")
    # print(model.feature_importances_)

    # print('r2', model.score(val_X, val_y))  ####r2 score
    # print('rmse', (mean_squared_error(val_y, y_pred4)) ** 0.5)

    # select random data to show
    i = random.randint(1, len(val_y) - 101)
    ind = np.linspace(0, 100, 100)

    plt.xlabel('Data index')
    plt.ylabel('Price in thousands of $')

    plt.plot(ind, np.expm1(val_y[i:i + 100]) / 1000, '-', linewidth=2.2, label='Test data')

    plt.plot(ind, np.expm1(y_pred[i:i + 100]) / 1000, '-.', linewidth=1.2, label='Random Forest Regression')
    plt.legend(loc='upper right')
    plt.xlim(-2, 112)

    ## feature importance, the higher the better
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features), reverse=True))

    plt.show()
    return model



# colSet = sales_data.columns

neighborhoodList = ['ALPHABET CITY', 'CHELSEA', 'CHINATOWN', 'CIVIC CENTER', 'CLINTON',
 'EAST VILLAGE', 'FASHION', 'FINANCIAL', 'FLATIRON', 'GRAMERCY',
 'GREENWICH VILLAGE-CENTRAL', 'GREENWICH VILLAGE-WEST', 'HARLEM-CENTRAL',
 'HARLEM-EAST', 'HARLEM-UPPER', 'HARLEM-WEST', 'INWOOD', 'JAVITS CENTER',
 'KIPS BAY', 'LITTLE ITALY', 'LOWER EAST SIDE', 'MANHATTAN VALLEY',
 'MIDTOWN CBD', 'MIDTOWN EAST', 'MIDTOWN WEST', 'MORNINGSIDE HEIGHTS',
 'MURRAY HILL', 'ROOSEVELT ISLAND', 'SOHO', 'SOUTHBRIDGE', 'TRIBECA',
 'UPPER EAST SIDE (59-79)', 'UPPER EAST SIDE (79-96)',
 'UPPER EAST SIDE (96-110)', 'UPPER WEST SIDE (59-79)',
 'UPPER WEST SIDE (79-96)', 'UPPER WEST SIDE (96-116)',
 'WASHINGTON HEIGHTS LOWER', 'WASHINGTON HEIGHTS UPPER', 'BATHGATE',
 'BAYCHESTER', 'BEDFORD PARK/NORWOOD', 'BELMONT', 'BRONX PARK', 'BRONXDALE',
 'CASTLE HILL/UNIONPORT', 'CITY ISLAND', 'CITY ISLAND-PELHAM STRIP',
 'CO-OP CITY', 'COUNTRY CLUB', 'CROTONA PARK', 'EAST RIVER', 'EAST TREMONT',
 'FIELDSTON', 'FORDHAM', 'HIGHBRIDGE/MORRIS HEIGHTS', 'HUNTS POINT',
 'KINGSBRIDGE HTS/UNIV HTS', 'KINGSBRIDGE/JEROME PARK', 'MELROSE/CONCOURSE',
 'MORRIS PARK/VAN NEST', 'MORRISANIA/LONGWOOD', 'MOTT HAVEN/PORT MORRIS',
 'MOUNT HOPE/MOUNT EDEN', 'PARKCHESTER', 'PELHAM BAY', 'PELHAM GARDENS',
 'PELHAM PARKWAY NORTH', 'PELHAM PARKWAY SOUTH', 'RIVERDALE',
 'SCHUYLERVILLE/PELHAM BAY', 'SOUNDVIEW', 'THROGS NECK', 'VAN CORTLANDT PARK',
 'WAKEFIELD', 'WESTCHESTER', 'WILLIAMSBRIDGE', 'WOODLAWN', 'BATH BEACH',
 'BAY RIDGE', 'BEDFORD STUYVESANT', 'BENSONHURST', 'BERGEN BEACH',
 'BOERUM HILL', 'BOROUGH PARK', 'BRIGHTON BEACH', 'BROOKLYN HEIGHTS',
 'BROWNSVILLE', 'BUSH TERMINAL', 'BUSHWICK', 'CANARSIE', 'CARROLL GARDENS',
 'CLINTON HILL', 'COBBLE HILL', 'COBBLE HILL-WEST', 'CONEY ISLAND',
 'CROWN HEIGHTS', 'CYPRESS HILLS', 'DOWNTOWN-FULTON FERRY',
 'DOWNTOWN-FULTON MALL', 'DOWNTOWN-METROTECH', 'DYKER HEIGHTS',
 'EAST NEW YORK', 'FLATBUSH-CENTRAL', 'FLATBUSH-EAST',
 'FLATBUSH-LEFFERTS GARDEN', 'FLATBUSH-NORTH', 'FLATLANDS', 'FORT GREENE',
 'GERRITSEN BEACH', 'GOWANUS', 'GRAVESEND', 'GREENPOINT', 'JAMAICA BAY',
 'KENSINGTON', 'MADISON', 'MANHATTAN BEACH', 'MARINE PARK', 'MIDWOOD',
 'MILL BASIN', 'NAVY YARD', 'OCEAN HILL', 'OCEAN PARKWAY-NORTH',
 'OCEAN PARKWAY-SOUTH', 'OLD MILL BASIN', 'PARK SLOPE', 'PARK SLOPE SOUTH',
 'PROSPECT HEIGHTS', 'RED HOOK', 'SEAGATE', 'SHEEPSHEAD BAY', 'SPRING CREEK',
 'SUNSET PARK', 'WILLIAMSBURG-CENTRAL', 'WILLIAMSBURG-EAST',
 'WILLIAMSBURG-NORTH', 'WILLIAMSBURG-SOUTH', 'WINDSOR TERRACE',
 'WYCKOFF HEIGHTS', 'AIRPORT LA GUARDIA', 'ARVERNE', 'ASTORIA', 'BAYSIDE',
 'BEECHHURST', 'BELLE HARBOR', 'BELLEROSE', 'BRIARWOOD', 'BROAD CHANNEL',
 'CAMBRIA HEIGHTS', 'COLLEGE POINT', 'CORONA', 'DOUGLASTON', 'EAST ELMHURST',
 'ELMHURST', 'FAR ROCKAWAY', 'FLORAL PARK', 'FLUSHING MEADOW PARK',
 'FLUSHING-NORTH', 'FLUSHING-SOUTH', 'FOREST HILLS', 'FRESH MEADOWS',
 'GLEN OAKS', 'GLENDALE', 'HAMMELS', 'HILLCREST', 'HOLLIS', 'HOLLIS HILLS',
 'HOLLISWOOD', 'HOWARD BEACH', 'JACKSON HEIGHTS', 'JAMAICA', 'JAMAICA ESTATES',
 'JAMAICA HILLS', 'KEW GARDENS', 'LAURELTON', 'LITTLE NECK',
 'LONG ISLAND CITY', 'MASPETH', 'MIDDLE VILLAGE', 'NEPONSIT',
 'OAKLAND GARDENS', 'OZONE PARK', 'QUEENS VILLAGE', 'REGO PARK',
 'RICHMOND HILL', 'RIDGEWOOD', 'ROCKAWAY PARK', 'ROSEDALE',
 'SO. JAMAICA-BAISLEY PARK', 'SOUTH JAMAICA', 'SOUTH OZONE PARK',
 'SPRINGFIELD GARDENS', 'ST. ALBANS', 'SUNNYSIDE', 'WHITESTONE', 'WOODHAVEN',
 'WOODSIDE', 'ANNADALE', 'ARDEN HEIGHTS', 'ARROCHAR', 'ARROCHAR-SHORE ACRES',
 'BLOOMFIELD', 'BULLS HEAD', 'CASTLETON CORNERS', 'CLOVE LAKES', 'CONCORD',
 'CONCORD-FOX HILLS', 'DONGAN HILLS', 'DONGAN HILLS-COLONY',
 'DONGAN HILLS-OLD TOWN', 'ELTINGVILLE', 'EMERSON HILL', 'FRESH KILLS',
 'GRANT CITY', 'GRASMERE', 'GREAT KILLS', 'GREAT KILLS-BAY TERRACE',
 'GRYMES HILL', 'HUGUENOT', 'LIVINGSTON', 'MANOR HEIGHTS', 'MARINERS HARBOR',
 'MIDLAND BEACH', 'NEW BRIGHTON', 'NEW BRIGHTON-ST. GEORGE', 'NEW DORP',
 'NEW DORP-BEACH', 'NEW DORP-HEIGHTS', 'NEW SPRINGVILLE', 'OAKWOOD',
 'OAKWOOD-BEACH', 'PLEASANT PLAINS', 'PORT IVORY', 'PORT RICHMOND',
 'PRINCES BAY', 'RICHMONDTOWN', 'RICHMONDTOWN-LIGHTHS HILL', 'ROSEBANK',
 'ROSSVILLE', 'ROSSVILLE-CHARLESTON', 'ROSSVILLE-PORT MOBIL',
 'ROSSVILLE-RICHMOND VALLEY', 'SILVER LAKE', 'SOUTH BEACH', 'STAPLETON',
 'STAPLETON-CLIFTON', 'TODT HILL', 'TOMPKINSVILLE', 'TOTTENVILLE', 'TRAVIS',
 'WEST NEW BRIGHTON', 'WESTERLEIGH', 'WILLOWBROOK', 'WOODROW']

neighborhood_map ={'ALPHABET CITY': 0, 'CHELSEA': 1, 'CHINATOWN': 2, 'CIVIC CENTER': 3, 'CLINTON': 4, 'EAST VILLAGE': 5, 'FASHION': 6, 'FINANCIAL': 7, 'FLATIRON': 8, 'GRAMERCY': 9, 'GREENWICH VILLAGE-CENTRAL': 10, 'GREENWICH VILLAGE-WEST': 11, 'HARLEM-CENTRAL': 12, 'HARLEM-EAST': 13, 'HARLEM-UPPER': 14, 'HARLEM-WEST': 15, 'INWOOD': 16, 'JAVITS CENTER': 17, 'KIPS BAY': 18, 'LITTLE ITALY': 19, 'LOWER EAST SIDE': 20, 'MANHATTAN VALLEY': 21, 'MIDTOWN CBD': 22, 'MIDTOWN EAST': 23, 'MIDTOWN WEST': 24, 'MORNINGSIDE HEIGHTS': 25, 'MURRAY HILL': 26, 'ROOSEVELT ISLAND': 27, 'SOHO': 28, 'SOUTHBRIDGE': 29, 'TRIBECA': 30, 'UPPER EAST SIDE (59-79)': 31, 'UPPER EAST SIDE (79-96)': 32, 'UPPER EAST SIDE (96-110)': 33, 'UPPER WEST SIDE (59-79)': 34, 'UPPER WEST SIDE (79-96)': 35, 'UPPER WEST SIDE (96-116)': 36, 'WASHINGTON HEIGHTS LOWER': 37, 'WASHINGTON HEIGHTS UPPER': 38, 'BATHGATE': 39, 'BAYCHESTER': 40, 'BEDFORD PARK/NORWOOD': 41, 'BELMONT': 42, 'BRONX PARK': 43, 'BRONXDALE': 44, 'CASTLE HILL/UNIONPORT': 45, 'CITY ISLAND': 46, 'CITY ISLAND-PELHAM STRIP': 47, 'CO-OP CITY': 48, 'COUNTRY CLUB': 49, 'CROTONA PARK': 50, 'EAST RIVER': 51, 'EAST TREMONT': 52, 'FIELDSTON': 53, 'FORDHAM': 54, 'HIGHBRIDGE/MORRIS HEIGHTS': 55, 'HUNTS POINT': 56, 'KINGSBRIDGE HTS/UNIV HTS': 57, 'KINGSBRIDGE/JEROME PARK': 58, 'MELROSE/CONCOURSE': 59, 'MORRIS PARK/VAN NEST': 60, 'MORRISANIA/LONGWOOD': 61, 'MOTT HAVEN/PORT MORRIS': 62, 'MOUNT HOPE/MOUNT EDEN': 63, 'PARKCHESTER': 64, 'PELHAM BAY': 65, 'PELHAM GARDENS': 66, 'PELHAM PARKWAY NORTH': 67, 'PELHAM PARKWAY SOUTH': 68, 'RIVERDALE': 69, 'SCHUYLERVILLE/PELHAM BAY': 70, 'SOUNDVIEW': 71, 'THROGS NECK': 72, 'VAN CORTLANDT PARK': 73, 'WAKEFIELD': 74, 'WESTCHESTER': 75, 'WILLIAMSBRIDGE': 76, 'WOODLAWN': 77, 'BATH BEACH': 78, 'BAY RIDGE': 79, 'BEDFORD STUYVESANT': 80, 'BENSONHURST': 81, 'BERGEN BEACH': 82, 'BOERUM HILL': 83, 'BOROUGH PARK': 84, 'BRIGHTON BEACH': 85, 'BROOKLYN HEIGHTS': 86, 'BROWNSVILLE': 87, 'BUSH TERMINAL': 88, 'BUSHWICK': 89, 'CANARSIE': 90, 'CARROLL GARDENS': 91, 'CLINTON HILL': 92, 'COBBLE HILL': 93, 'COBBLE HILL-WEST': 94, 'CONEY ISLAND': 95, 'CROWN HEIGHTS': 96, 'CYPRESS HILLS': 97, 'DOWNTOWN-FULTON FERRY': 98, 'DOWNTOWN-FULTON MALL': 99, 'DOWNTOWN-METROTECH': 100, 'DYKER HEIGHTS': 101, 'EAST NEW YORK': 102, 'FLATBUSH-CENTRAL': 103, 'FLATBUSH-EAST': 104, 'FLATBUSH-LEFFERTS GARDEN': 105, 'FLATBUSH-NORTH': 106, 'FLATLANDS': 107, 'FORT GREENE': 108, 'GERRITSEN BEACH': 109, 'GOWANUS': 110, 'GRAVESEND': 111, 'GREENPOINT': 112, 'JAMAICA BAY': 113, 'KENSINGTON': 114, 'MADISON': 115, 'MANHATTAN BEACH': 116, 'MARINE PARK': 117, 'MIDWOOD': 118, 'MILL BASIN': 119, 'NAVY YARD': 120, 'OCEAN HILL': 121, 'OCEAN PARKWAY-NORTH': 122, 'OCEAN PARKWAY-SOUTH': 123, 'OLD MILL BASIN': 124, 'PARK SLOPE': 125, 'PARK SLOPE SOUTH': 126, 'PROSPECT HEIGHTS': 127, 'RED HOOK': 128, 'SEAGATE': 129, 'SHEEPSHEAD BAY': 130, 'SPRING CREEK': 131, 'SUNSET PARK': 132, 'WILLIAMSBURG-CENTRAL': 133, 'WILLIAMSBURG-EAST': 134, 'WILLIAMSBURG-NORTH': 135, 'WILLIAMSBURG-SOUTH': 136, 'WINDSOR TERRACE': 137, 'WYCKOFF HEIGHTS': 138, 'AIRPORT LA GUARDIA': 139, 'ARVERNE': 140, 'ASTORIA': 141, 'BAYSIDE': 142, 'BEECHHURST': 143, 'BELLE HARBOR': 144, 'BELLEROSE': 145, 'BRIARWOOD': 146, 'BROAD CHANNEL': 147, 'CAMBRIA HEIGHTS': 148, 'COLLEGE POINT': 149, 'CORONA': 150, 'DOUGLASTON': 151, 'EAST ELMHURST': 152, 'ELMHURST': 153, 'FAR ROCKAWAY': 154, 'FLORAL PARK': 155, 'FLUSHING MEADOW PARK': 156, 'FLUSHING-NORTH': 157, 'FLUSHING-SOUTH': 158, 'FOREST HILLS': 159, 'FRESH MEADOWS': 160, 'GLEN OAKS': 161, 'GLENDALE': 162, 'HAMMELS': 163, 'HILLCREST': 164, 'HOLLIS': 165, 'HOLLIS HILLS': 166, 'HOLLISWOOD': 167, 'HOWARD BEACH': 168, 'JACKSON HEIGHTS': 169, 'JAMAICA': 170, 'JAMAICA ESTATES': 171, 'JAMAICA HILLS': 172, 'KEW GARDENS': 173, 'LAURELTON': 174, 'LITTLE NECK': 175, 'LONG ISLAND CITY': 176, 'MASPETH': 177, 'MIDDLE VILLAGE': 178, 'NEPONSIT': 179, 'OAKLAND GARDENS': 180, 'OZONE PARK': 181, 'QUEENS VILLAGE': 182, 'REGO PARK': 183, 'RICHMOND HILL': 184, 'RIDGEWOOD': 185, 'ROCKAWAY PARK': 186, 'ROSEDALE': 187, 'SO. JAMAICA-BAISLEY PARK': 188, 'SOUTH JAMAICA': 189, 'SOUTH OZONE PARK': 190, 'SPRINGFIELD GARDENS': 191, 'ST. ALBANS': 192, 'SUNNYSIDE': 193, 'WHITESTONE': 194, 'WOODHAVEN': 195, 'WOODSIDE': 196, 'ANNADALE': 197, 'ARDEN HEIGHTS': 198, 'ARROCHAR': 199, 'ARROCHAR-SHORE ACRES': 200, 'BLOOMFIELD': 201, 'BULLS HEAD': 202, 'CASTLETON CORNERS': 203, 'CLOVE LAKES': 204, 'CONCORD': 205, 'CONCORD-FOX HILLS': 206, 'DONGAN HILLS': 207, 'DONGAN HILLS-COLONY': 208, 'DONGAN HILLS-OLD TOWN': 209, 'ELTINGVILLE': 210, 'EMERSON HILL': 211, 'FRESH KILLS': 212, 'GRANT CITY': 213, 'GRASMERE': 214, 'GREAT KILLS': 215, 'GREAT KILLS-BAY TERRACE': 216, 'GRYMES HILL': 217, 'HUGUENOT': 218, 'LIVINGSTON': 219, 'MANOR HEIGHTS': 220, 'MARINERS HARBOR': 221, 'MIDLAND BEACH': 222, 'NEW BRIGHTON': 223, 'NEW BRIGHTON-ST. GEORGE': 224, 'NEW DORP': 225, 'NEW DORP-BEACH': 226, 'NEW DORP-HEIGHTS': 227, 'NEW SPRINGVILLE': 228, 'OAKWOOD': 229, 'OAKWOOD-BEACH': 230, 'PLEASANT PLAINS': 231, 'PORT IVORY': 232, 'PORT RICHMOND': 233, 'PRINCES BAY': 234, 'RICHMONDTOWN': 235, 'RICHMONDTOWN-LIGHTHS HILL': 236, 'ROSEBANK': 237, 'ROSSVILLE': 238, 'ROSSVILLE-CHARLESTON': 239, 'ROSSVILLE-PORT MOBIL': 240, 'ROSSVILLE-RICHMOND VALLEY': 241, 'SILVER LAKE': 242, 'SOUTH BEACH': 243, 'STAPLETON': 244, 'STAPLETON-CLIFTON': 245, 'TODT HILL': 246, 'TOMPKINSVILLE': 247, 'TOTTENVILLE': 248, 'TRAVIS': 249, 'WEST NEW BRIGHTON': 250, 'WESTERLEIGH': 251, 'WILLOWBROOK': 252, 'WOODROW': 253}


index = 0
building_class_list = ['07rentals_walkupapartments', '08rentals_elevatorapartments',
 '09coops_walkupapartments', '10coops_elevatorapartments',
 '11acondo_rentals', '12condos_walkupapartments',
 '13condos_elevatorapartments', '14rentals_4_10unit',
 '15condos_2_10unitresidential', '16condos_2_10unitwithcommercialunit',
 '17condocoops', '22storebuildings', '37religiousfacilities',
 '42condocultural/medical/educational/etc', '46condostorebuildings',
 '47condonon_businessstorage', '01onefamilydwellings',
 '02twofamilydwellings', '03threefamilydwellings', '04taxclass1condos',
 '21officebuildings', '23loftbuildings', '25luxuryhotels', '26otherhotels',
 '28commercialcondos', '29commercialgarages',
 '35indoorpublicandculturalfacilities', '38asylumsandhomes',
 '43condoofficebuildings', '44condoparking',
 '48condoterraces/gardens/cabanas', '31commercialvacantland',
 '32hospitalandhealthfacilities', '41taxclass4_other',
 '18taxclass3_untilityproperties', '30warehouses',
 '36outdoorrecreationalfacilities', '49condowarehouses/factory/indus',
 '34theatres', '27factories', '40selectedgovernmentalfacilities',
 '45condohotels', '33educationalfacilities', '11specialcondobillinglots',
 '05taxclass1vacantland', '06taxclass1_other', '39transportationfacilities']

building_class_map ={'07rentals_walkupapartments': 0, '08rentals_elevatorapartments': 1, '09coops_walkupapartments': 2, '10coops_elevatorapartments': 3, '11acondo_rentals': 4, '12condos_walkupapartments': 5, '13condos_elevatorapartments': 6, '14rentals_4_10unit': 7, '15condos_2_10unitresidential': 8, '16condos_2_10unitwithcommercialunit': 9, '17condocoops': 10, '22storebuildings': 11, '37religiousfacilities': 12, '42condocultural/medical/educational/etc': 13, '46condostorebuildings': 14, '47condonon_businessstorage': 15, '01onefamilydwellings': 16, '02twofamilydwellings': 17, '03threefamilydwellings': 18, '04taxclass1condos': 19, '21officebuildings': 20, '23loftbuildings': 21, '25luxuryhotels': 22, '26otherhotels': 23, '28commercialcondos': 24, '29commercialgarages': 25, '35indoorpublicandculturalfacilities': 26, '38asylumsandhomes': 27, '43condoofficebuildings': 28, '44condoparking': 29, '48condoterraces/gardens/cabanas': 30, '31commercialvacantland': 31, '32hospitalandhealthfacilities': 32, '41taxclass4_other': 33, '18taxclass3_untilityproperties': 34, '30warehouses': 35, '36outdoorrecreationalfacilities': 36, '49condowarehouses/factory/indus': 37, '34theatres': 38, '27factories': 39, '40selectedgovernmentalfacilities': 40, '45condohotels': 41, '33educationalfacilities': 42, '11specialcondobillinglots': 43, '05taxclass1vacantland': 44, '06taxclass1_other': 45, '39transportationfacilities': 46}

tax_class_at_present_list = ['2A', '2', '2B', '2C', ' ', '4', '1', '1C', '3', '1A', '1B']
tax_class_at_present_map = {'2A': 0, '2': 1, '2B': 2, '2C': 3, ' ': 4, '4': 5, '1': 6, '1C': 7, '3': 8, '1A': 9, '1B': 10}

building_class_at_present_list = ['C2', 'C7', 'C4', 'D5', 'D9', 'D7', 'D1', 'C6', 'D0', 'D4', 'RR', ' ', 'R2', 'R4', 'S3',
 'S4', 'S5', 'R1', 'R8', 'R9', 'K4', 'M9', 'M3', 'RK', 'RS', 'A9', 'A4', 'B3', 'B1',
 'S2', 'C0', 'R6', 'C5', 'C3', 'C1', 'D6', 'S9', 'O2', 'O1', 'O3', 'O5', 'O6', 'K1',
 'K2', 'L9', 'L8', 'L1', 'H1', 'H8', 'H3', 'R5', 'G6', 'P7', 'M1', 'N2', 'RB', 'RG',
 'RT', 'K9', 'V1', 'GW', 'G2', 'I7', 'M4', 'Z9', 'B9', 'D3', 'G9', 'I9', 'U6', 'O4',
 'L3', 'H2', 'E1', 'Z3', 'RW', 'C9', 'J5', 'N9', 'S1', 'A5', 'J8', 'B2', 'C8', 'F5',
 'Q1', 'G7', 'G5', 'G4', 'P2', 'Q9', 'Y1', 'RA', 'RP', 'O8', 'HR', 'G1', 'E7', 'I5',
 'R3', 'I4', 'H9', 'RH', 'D8', 'HB', 'J4', 'W2', 'P9', 'A7', 'D2', 'S0', 'O7', 'O9',
 'W3', 'HS', 'H6', 'J9', 'R0', 'HH', 'W8', 'W6', 'A1', 'K5', 'F1', 'V9', 'A2', 'V0',
 'G0', 'F4', 'E9', 'I3', 'W4', 'V3', 'I1', 'A6', 'Q8', 'A3', 'Z0', 'W1', 'U1', 'F2',
 'F9', 'GU', 'I6', 'G8', 'P5', 'Y3', 'W9', 'M2', 'G3', 'V6', 'K7', 'K3', 'R7', 'P8',
 'K6', 'V2', 'E2', 'Z2', 'T2', 'K8', 'P6', 'A0', 'H4', 'J1', 'CM', 'Z7']
building_class_at_present_map = {'C2': 0, 'C7': 1, 'C4': 2, 'D5': 3, 'D9': 4, 'D7': 5, 'D1': 6, 'C6': 7, 'D0': 8, 'D4': 9, 'RR': 10, ' ': 11, 'R2': 12, 'R4': 13, 'S3': 14, 'S4': 15, 'S5': 16, 'R1': 17, 'R8': 18, 'R9': 19, 'K4': 20, 'M9': 21, 'M3': 22, 'RK': 23, 'RS': 24, 'A9': 25, 'A4': 26, 'B3': 27, 'B1': 28, 'S2': 29, 'C0': 30, 'R6': 31, 'C5': 32, 'C3': 33, 'C1': 34, 'D6': 35, 'S9': 36, 'O2': 37, 'O1': 38, 'O3': 39, 'O5': 40, 'O6': 41, 'K1': 42, 'K2': 43, 'L9': 44, 'L8': 45, 'L1': 46, 'H1': 47, 'H8': 48, 'H3': 49, 'R5': 50, 'G6': 51, 'P7': 52, 'M1': 53, 'N2': 54, 'RB': 55, 'RG': 56, 'RT': 57, 'K9': 58, 'V1': 59, 'GW': 60, 'G2': 61, 'I7': 62, 'M4': 63, 'Z9': 64, 'B9': 65, 'D3': 66, 'G9': 67, 'I9': 68, 'U6': 69, 'O4': 70, 'L3': 71, 'H2': 72, 'E1': 73, 'Z3': 74, 'RW': 75, 'C9': 76, 'J5': 77, 'N9': 78, 'S1': 79, 'A5': 80, 'J8': 81, 'B2': 82, 'C8': 83, 'F5': 84, 'Q1': 85, 'G7': 86, 'G5': 87, 'G4': 88, 'P2': 89, 'Q9': 90, 'Y1': 91, 'RA': 92, 'RP': 93, 'O8': 94, 'HR': 95, 'G1': 96, 'E7': 97, 'I5': 98, 'R3': 99, 'I4': 100, 'H9': 101, 'RH': 102, 'D8': 103, 'HB': 104, 'J4': 105, 'W2': 106, 'P9': 107, 'A7': 108, 'D2': 109, 'S0': 110, 'O7': 111, 'O9': 112, 'W3': 113, 'HS': 114, 'H6': 115, 'J9': 116, 'R0': 117, 'HH': 118, 'W8': 119, 'W6': 120, 'A1': 121, 'K5': 122, 'F1': 123, 'V9': 124, 'A2': 125, 'V0': 126, 'G0': 127, 'F4': 128, 'E9': 129, 'I3': 130, 'W4': 131, 'V3': 132, 'I1': 133, 'A6': 134, 'Q8': 135, 'A3': 136, 'Z0': 137, 'W1': 138, 'U1': 139, 'F2': 140, 'F9': 141, 'GU': 142, 'I6': 143, 'G8': 144, 'P5': 145, 'Y3': 146, 'W9': 147, 'M2': 148, 'G3': 149, 'V6': 150, 'K7': 151, 'K3': 152, 'R7': 153, 'P8': 154, 'K6': 155, 'V2': 156, 'E2': 157, 'Z2': 158, 'T2': 159, 'K8': 160, 'P6': 161, 'A0': 162, 'H4': 163, 'J1': 164, 'CM': 165, 'Z7': 166}

building_class_at_time_of_sale_list = ['C2', 'C7', 'C4', 'D5', 'D9', 'D7', 'D1', 'C6', 'D0', 'D4', 'RR', 'R2', 'R4', 'S3',
 'S4', 'S5', 'R1', 'R8', 'R9', 'K4', 'M9', 'M3', 'RA', 'RK', 'RS', 'A9', 'A4', 'B3',
 'B1', 'S2', 'C0', 'R6', 'C5', 'C3', 'C1', 'D6', 'S9', 'O2', 'O1', 'O3', 'O5', 'O6',
 'K1', 'K2', 'L9', 'L8', 'L1', 'H1', 'H8', 'H3', 'R5', 'G9', 'G6', 'P7', 'M1', 'N2',
 'RB', 'RG', 'RT', 'K9', 'V1', 'GW', 'G2', 'I7', 'M4', 'Z9', 'B9', 'D3', 'I9', 'U6',
 'O4', 'L3', 'H2', 'E1', 'Z3', 'Q1', 'RW', 'C9', 'J5', 'N9', 'S1', 'A5', 'J8', 'B2',
 'C8', 'F5', 'G7', 'G5', 'G4', 'P2', 'Q9', 'Y1', 'RP', 'O8', 'HR', 'G1', 'E7', 'I5',
 'R3', 'I4', 'H9', 'RH', 'D8', 'HB', 'J4', 'W2', 'P9', 'A7', 'D2', 'S0', 'O7', 'O9',
 'W3', 'HS', 'H6', 'J9', 'R0', 'HH', 'W8', 'W6', 'A1', 'K5', 'F1', 'V9', 'A2', 'V0',
 'G0', 'F4', 'E9', 'I3', 'W4', 'V3', 'I1', 'A6', 'Q8', 'A3', 'Z0', 'W1', 'U1', 'F2',
 'F9', 'GU', 'I6', 'G8', 'P5', 'Y3', 'W9', 'M2', 'G3', 'V6', 'K7', 'K3', 'H4', 'R7',
 'P8', 'K6', 'V2', 'E2', 'Z2', 'T2', 'K8', 'P6', 'A0', 'J1', 'CM', 'Z7']

building_class_at_time_of_sale_map = {'C2': 0, 'C7': 1, 'C4': 2, 'D5': 3, 'D9': 4, 'D7': 5, 'D1': 6, 'C6': 7, 'D0': 8, 'D4': 9, 'RR': 10, 'R2': 11, 'R4': 12, 'S3': 13, 'S4': 14, 'S5': 15, 'R1': 16, 'R8': 17, 'R9': 18, 'K4': 19, 'M9': 20, 'M3': 21, 'RA': 22, 'RK': 23, 'RS': 24, 'A9': 25, 'A4': 26, 'B3': 27, 'B1': 28, 'S2': 29, 'C0': 30, 'R6': 31, 'C5': 32, 'C3': 33, 'C1': 34, 'D6': 35, 'S9': 36, 'O2': 37, 'O1': 38, 'O3': 39, 'O5': 40, 'O6': 41, 'K1': 42, 'K2': 43, 'L9': 44, 'L8': 45, 'L1': 46, 'H1': 47, 'H8': 48, 'H3': 49, 'R5': 50, 'G9': 51, 'G6': 52, 'P7': 53, 'M1': 54, 'N2': 55, 'RB': 56, 'RG': 57, 'RT': 58, 'K9': 59, 'V1': 60, 'GW': 61, 'G2': 62, 'I7': 63, 'M4': 64, 'Z9': 65, 'B9': 66, 'D3': 67, 'I9': 68, 'U6': 69, 'O4': 70, 'L3': 71, 'H2': 72, 'E1': 73, 'Z3': 74, 'Q1': 75, 'RW': 76, 'C9': 77, 'J5': 78, 'N9': 79, 'S1': 80, 'A5': 81, 'J8': 82, 'B2': 83, 'C8': 84, 'F5': 85, 'G7': 86, 'G5': 87, 'G4': 88, 'P2': 89, 'Q9': 90, 'Y1': 91, 'RP': 92, 'O8': 93, 'HR': 94, 'G1': 95, 'E7': 96, 'I5': 97, 'R3': 98, 'I4': 99, 'H9': 100, 'RH': 101, 'D8': 102, 'HB': 103, 'J4': 104, 'W2': 105, 'P9': 106, 'A7': 107, 'D2': 108, 'S0': 109, 'O7': 110, 'O9': 111, 'W3': 112, 'HS': 113, 'H6': 114, 'J9': 115, 'R0': 116, 'HH': 117, 'W8': 118, 'W6': 119, 'A1': 120, 'K5': 121, 'F1': 122, 'V9': 123, 'A2': 124, 'V0': 125, 'G0': 126, 'F4': 127, 'E9': 128, 'I3': 129, 'W4': 130, 'V3': 131, 'I1': 132, 'A6': 133, 'Q8': 134, 'A3': 135, 'Z0': 136, 'W1': 137, 'U1': 138, 'F2': 139, 'F9': 140, 'GU': 141, 'I6': 142, 'G8': 143, 'P5': 144, 'Y3': 145, 'W9': 146, 'M2': 147, 'G3': 148, 'V6': 149, 'K7': 150, 'K3': 151, 'H4': 152, 'R7': 153, 'P8': 154, 'K6': 155, 'V2': 156, 'E2': 157, 'Z2': 158, 'T2': 159, 'K8': 160, 'P6': 161, 'A0': 162, 'J1': 163, 'CM': 164, 'Z7': 165}

# # create map
# index = 0
# for n in building_class_at_time_of_sale_list:
#     building_class_at_time_of_sale_map[n] = index
#     index = index+1
#     # print(building_class_map)
# print(building_class_at_time_of_sale_map)
# sys.exit()

# for i in colSet:
#
#     values = sales_data.get(i).unique()
#     print("====")
#     print(i)
#     print(values)
# sys.exit()

def convertToInt(str, map):
   if(str in map):
       return map[str]
   else:
       return -1


def sale(string):
    if ('-' in string):
        return 0
    else:
        return int(string)
    
def address(string):
    if ('-' in string):
        return 0
    else:
        return (string + " NYC")

def year_built(string):
    if string==0:
        return 1000
    elif ('-' in string):
        return 1000
    else:
        return int(string)


def predict(model):
    print()
    print("#predict")

    land_square = 8
    gross_square = 9
    year_built = 2000
    district  = 2
    neighborhood = 0
    tax_class_at_present = 2
    building_class_category = 3
    building_class_at_present = 2
    building_class_at_time_of_sale = 3
    residential_units = 10
    commercial_units = 0
    total_units = 12

    predict_X = [[land_square,gross_square, year_built,district, neighborhood, tax_class_at_present, building_class_category, building_class_at_present, building_class_at_time_of_sale,
                  residential_units,    commercial_units,  total_units ]]

    predict_y = model.predict(predict_X)


    print("Predicted price is " + str(np.expm1(predict_y)))




if __name__ == '__main__':
    data = load_data()
    analyze_data(data)
    data = cleanData(data)
    data.to_csv('cleaned_house_data.csv')
    data_engineer(data)
    model = build_model(data)
    predict(model)
    pass

    import geopandas
    boros = geopandas.read_file("cleaned_house_data4.csv", encoding='utf8')
    boro_locations = geopandas.tools.geocode(boros.address)
    boro_locations.to_file("geography.shp")
    
    from geopandas.tools import geocode
    key = 'AIzaSyCXp2Qhw4BGaeiagHNchQw0yPGMKH_XTko'
    geo = geocode(boros['address'], api_key=key)
    
    with open('cleaned_house_data3.csv') as f:
        print (f)
        
       
        
    import pandas
    data = pandas.read_csv("cleaned_house_data4.csv", encoding='utf8')    
    import googlemaps
    gmaps_key = googlemaps.Client(key='Deleted4Security')
    data['LAT'] = None
    data['LON'] = None
    for i in range(0, len(data), 1):
        geocode_result = gmaps_key.geocode(data['address'][i])
    


