import pandas as pd
import requests
import json
import os
from io import StringIO


url = "https://ncataggies.com/sports/mens-basketball/stats/2025-26"
output_file = "assets/data/stats.json"


def clean_player_name(name):
    return " ".join(str(name).split())

try:
    print(f"Fetching data from {url}...")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    
    all_tables = pd.read_html(StringIO(response.text), header=[0, 1])
    # Print how many tables pandas found hidden in the HTML code
    print(f"Found {len(all_tables)} tables. Processing...")

    # Create an empty dictionary that will hold our final, cleaned datasets
    datasets = {}

    # Start a loop to check every single table pandas found, one by one
    for i, original_df in enumerate(all_tables):
        
        # Make a safe copy of the table so we don't accidentally edit the original downloaded version
        df = original_df.copy()
        
        # --- 3. CLEAN & MERGE MULTI-LEVEL HEADERS ---
        
        # Create an empty list to hold our new, flattened column names
        new_cols = []
        
        # Loop through the pairs of top/bottom headers (e.g., 'Rebounds' and 'OFF')
        for col in df.columns.values:
            # Clean up the top header and remove surrounding spaces
            c1 = str(col[0]).strip()
            # Clean up the bottom header and remove surrounding spaces
            c2 = str(col[1]).strip()
            
            # If the top header is totally blank or named 'nan', just use the bottom header
            if "Unnamed" in c1 or c1.lower() == 'nan' or c1 == '': 
                col_name = c2
            # If the bottom header is totally blank or named 'nan', just use the top header
            elif "Unnamed" in c2 or c2.lower() == 'nan' or c2 == '': 
                col_name = c1
            # THE FIX: If the top and bottom header are the exact same word (like "AST", "AST"), just use the word once
            elif c1 == c2: 
                col_name = c1 
            # Otherwise, combine them together with an underscore (e.g., "Rebounds_OFF")
            else: 
                col_name = f"{c1}_{c2}" 
                
            # Add our new clean column name to the list
            new_cols.append(col_name.strip())
            
        # Replace the messy table headers with our clean list of names
        df.columns = new_cols
        
        # Start a loop to make sure no two columns have the exact same name (which crashes pandas)
        counts = {}
        unique_cols = []
        for col in df.columns:
            # If we haven't seen this column name yet, mark its count as 0
            if col not in counts:
                counts[col] = 0
                unique_cols.append(col)
            # If we HAVE seen it before, add a number to the end of it (e.g., "AVG_1")
            else:
                counts[col] += 1
                unique_cols.append(f"{col}_{counts[col]}")
        # Apply the unique names to the table
        df.columns = unique_cols

        # VALIDATION: Check if the word "Player" exists in the table's column names
        if not any("Player" in c for c in df.columns):
            # If "Player" isn't there, this is a junk table (like a schedule). Skip to the next table.
            continue
            
        # Print a debug message saying we found a valid basketball table
        print(f"\n--- Found a Player Stats Table ---")

        # --- 4. ADVANCED COLUMN MAPPING ---
        
        # Define a flexible function to hunt for the right columns
        def get_col(df_columns, exact_names):
            # First, check if the exact name exists (case-insensitive)
            for col in df_columns:
                for name in exact_names:
                    if col.upper().strip() == name.upper().strip():
                        return col
            # If not found, check if the keyword is at least a substring inside the column name
            for col in df_columns:
                for name in exact_names:
                    if name.upper().strip() in col.upper().strip():
                        return col
            # If it totally fails, return None
            return None

        # Create a dictionary to link our standard stat names to whatever the website called them
        col_map = {}
        # Find Player and Jersey Number
        col_map['Player'] = get_col(df.columns, ['Player', 'Name'])
        col_map['Jersey'] = get_col(df.columns, ['#', 'No.'])
        
        # Find Games Played and Games Started
        col_map['GP'] = get_col(df.columns, ['GP', 'MINUTES_GP'])
        col_map['GS'] = get_col(df.columns, ['GS', 'MINUTES_GS'])
        
        # Find Minutes
        col_map['MIN'] = get_col(df.columns, ['Minutes_TOT', 'MIN'])
        
        # Find Field Goals
        col_map['FGM'] = get_col(df.columns, ['FG_FGM', 'FGM'])
        col_map['FGA'] = get_col(df.columns, ['FG_FGA', 'FGA'])
        col_map['FG_PCT'] = get_col(df.columns, ['FG_FG%', 'FG%'])
        
        # Find 3-Pointers
        col_map['FG3M'] = get_col(df.columns, ['3PT_3PT', '3PT'])
        col_map['FG3A'] = get_col(df.columns, ['3PT_3PTA', '3PTA'])
        col_map['FG3_PCT'] = get_col(df.columns, ['3PT_3PT%', '3PT%'])
        
        # Find Free Throws
        col_map['FTM'] = get_col(df.columns, ['FT_FTM', 'FTM'])
        col_map['FTA'] = get_col(df.columns, ['FT_FTA', 'FTA'])
        col_map['FT_PCT'] = get_col(df.columns, ['FT_FT%', 'FT%'])
        
        # Find Scoring
        col_map['PTS'] = get_col(df.columns, ['Scoring_PTS', 'PTS'])
        
        # Find Rebounds (Offensive, Defensive, Total)
        col_map['OREB'] = get_col(df.columns, ['Rebounds_OFF', 'OREB', 'OFF'])
        col_map['DREB'] = get_col(df.columns, ['Rebounds_DEF', 'DREB', 'DEF'])
        col_map['REB']  = get_col(df.columns, ['Rebounds_TOT', 'REB', 'TOT'])
        
        # Find Playmaking and Defense
        col_map['PF']  = get_col(df.columns, ['PF', 'FOULS'])
        col_map['AST'] = get_col(df.columns, ['AST', 'ASSISTS'])
        col_map['TO']  = get_col(df.columns, ['TO', 'TURNOVERS'])
        col_map['STL'] = get_col(df.columns, ['STL', 'STEALS'])
        col_map['BLK'] = get_col(df.columns, ['BLK', 'BLOCKS'])

        # If the table is missing "Player" or "Points", it's missing critical data, so skip it
        if not col_map['Player'] or not col_map['PTS']:
            print("-> Missing Player or PTS column. Skipping table.")
            continue

        # --- 5. EXTRACT DATA ROW BY ROW ---
        
        # Create an empty list to hold the cleaned player data objects
        final_rows = []
        # Create a variable to track the highest score in the table (helps us identify table types)
        max_pts = 0
        
        # Loop through every single row (player) in the table
        for _, row in df.iterrows():
            try:
                # Grab the text in the player name column
                p_name = row[col_map['Player']]
                # Skip header rows that get repeated in the middle of the table by Sidearm Sports
                if str(p_name) in ["Player", "Team", "Total", "Opponents"]: continue
                # Skip empty blank rows
                if pd.isna(p_name): continue

                # A helper function to safely pull out numeric data, turning missing data into 0.0
                def get_val(key):
                    c = col_map.get(key)
                    if c and c in row:
                        val = pd.to_numeric(row[c], errors='coerce')
                        return 0.0 if pd.isna(val) else val
                    return 0.0

                # Pull out Games Played
                gp = int(get_val('GP'))
                # If the player hasn't played any games, skip them completely
                if gp == 0: continue 

                # Pull out Total Points
                pts = float(get_val('PTS'))
                # Update our max points tracker if this player scored the most so far
                if pts > max_pts: max_pts = pts

                # Build the final dictionary object containing all the scraped numbers for this player
                player_obj = {
                    "Player": clean_player_name(p_name),
                    "#": str(row.get(col_map['Jersey'], '00')).replace('.0',''),
                    "GP": gp,
                    "GS": int(get_val('GS')),
                    "MIN": float(get_val('MIN')),
                    "FGM": float(get_val('FGM')),
                    "FGA": float(get_val('FGA')),
                    "FG_PCT": float(get_val('FG_PCT')),
                    "FG3M": float(get_val('FG3M')),
                    "FG3A": float(get_val('FG3A')),
                    "FG3_PCT": float(get_val('FG3_PCT')),
                    "FTM": float(get_val('FTM')),
                    "FTA": float(get_val('FTA')),
                    "FT_PCT": float(get_val('FT_PCT')),
                    "PTS": pts,
                    "OREB": float(get_val('OREB')),
                    "DREB": float(get_val('DREB')),
                    "REB": float(get_val('REB')),
                    "PF": float(get_val('PF')),
                    "AST": float(get_val('AST')),
                    "TO": float(get_val('TO')),
                    "STL": float(get_val('STL')),
                    "BLK": float(get_val('BLK')),
                }

                # Calculate Per-Game Averages manually by dividing their Totals by Games Played
                player_obj['AVG_MIN'] = round(player_obj['MIN'] / gp, 1)
                player_obj['AVG_PTS'] = round(player_obj['PTS'] / gp, 1)
                player_obj['AVG_REB'] = round(player_obj['REB'] / gp, 1)
                player_obj['AVG_AST'] = round(player_obj['AST'] / gp, 1)
                player_obj['AVG_TO']  = round(player_obj['TO'] / gp, 1)
                player_obj['AVG_STL'] = round(player_obj['STL'] / gp, 1)
                player_obj['AVG_BLK'] = round(player_obj['BLK'] / gp, 1)
                player_obj['AVG_PF']  = round(player_obj['PF'] / gp, 1)

                # Add this player's finished dictionary to our list of rows
                final_rows.append(player_obj)
                
            # If a specific player row causes an error, ignore it and continue to the next player
            except Exception as e:
                continue
        
        # If the table ended up empty after all that, skip to the next table
        if not final_rows: continue

        # --- 6. SAVE TO DATASET ---
        
        # If max points > 50, this is a "Totals" table.
        # THE BIG FIX: Only save it if we haven't already saved "season_totals" to prevent overwriting!
        if max_pts > 50 and "season_totals" not in datasets:
            
            print("-> Success! Captured the MAIN Overall Totals table.")
            
            # Save the raw numbers dataset for your advanced charts
            datasets["season_totals"] = {
                "name": "Season Totals (Live)",
                "players": final_rows,
                # These strings become the interactive buttons on your dashboard
                "stats": ["PTS", "REB", "AST", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "STL", "BLK", "TO", "PF"]
            }
            
            # Save the mathematically calculated averages dataset as the default view
            datasets["men_season_avg"] = {
                "name": "Men's Season Averages (Live)",
                "players": final_rows,
                "stats": ["AVG_PTS", "AVG_REB", "AVG_AST", "AVG_MIN", "AVG_STL", "AVG_BLK", "AVG_TO", "AVG_PF", "FG_PCT", "FG3_PCT", "FT_PCT", "OREB", "DREB"]
            }

    # EXPORT TO JSON 
    
    # Generate a timestamp so your dashboard knows when the data was scraped
    datasets["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the "assets/data/" folder exists; if it doesn't, automatically create it
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open the output file in write mode ('w')
    with open(output_file, 'w') as f:
        # Convert our Python dictionary into a JSON string and save it to the file
        json.dump(datasets, f, indent=4)

    # Print a final success message indicating the script is finished
    print(f"\nFile successfully saved to {output_file}.")

# If the whole script crashes, print the error instead of silently failing
except Exception as e:
    print(f"Error scraping data: {e}")