import os

import typer

from src.collection.building import collect_building
from src.collection.gtfs import collect_gtfs
from src.collection.landuse import collect_landuse
from src.collection.network import collect_network
from src.collection.osm_pt_lines import collect_osm_pt_lines
from src.collection.overture import collect_overture
from src.collection.poi import collect_poi
from src.core.config import settings
from src.db.db import Database
from src.export.gtfs import export_gtfs
from src.fusion.poi_osm_overture import fusion_poi_osm_overture
from src.migration.gtfs import migrate_gtfs
from src.preparation.building import prepare_building
from src.preparation.gtfs import prepare_gtfs
from src.preparation.gtfs_stations import prepare_gtfs_stations
from src.preparation.gtfs_stops import prepare_gtfs_stops
from src.preparation.network import export_network, prepare_network
from src.preparation.network_pt import prepare_network_pt
from src.preparation.osm_pt_lines import prepare_osm_pt_lines
from src.preparation.overture_division_area import prepare_overture_division_area
from src.preparation.overture_place import prepare_overture_place
from src.preparation.overture_street_network import prepare_overture_street_network
from src.preparation.poi import export_poi, prepare_poi
from src.preparation.poi_overture import prepare_poi_overture
from src.preparation.population import prepare_population
from src.utils.utils import print_hashtags, print_info
from src.validation.poi import validate_poi

app = typer.Typer()

db = Database(settings.LOCAL_DATABASE_URI)
db_rd = Database(settings.RAW_DATABASE_URI)

# TODO: Add prepare_landuse, export_building, export_landuse, export_population
action_dict = {
    "collection": {
        "building": collect_building,
        "poi": collect_poi,
        "landuse": collect_landuse,
        "network": collect_network,
        "gtfs": collect_gtfs,
        "overture": collect_overture,
        "osm_pt_lines": collect_osm_pt_lines,
    },
    "preparation": {
        "poi": prepare_poi,
        "poi_overture": prepare_poi_overture,
        "network": prepare_network,
        "network_pt": prepare_network_pt,
        "building": prepare_building,
        "population": prepare_population,
        "gtfs": prepare_gtfs,
        "overture_street_network": prepare_overture_street_network,
        "overture": prepare_overture_division_area,
        "gtfs_stops": prepare_gtfs_stops,
        "gtfs_stations": prepare_gtfs_stations,
        "osm_pt_lines": prepare_osm_pt_lines,
        "overture_place": prepare_overture_place,
    },
    "fusion":{
        "poi_osm_overture": fusion_poi_osm_overture,
    },
    "export": {
        "poi": export_poi,
        "network": export_network,
        "gtfs": export_gtfs,
    },
    "migration": {
        "gtfs": migrate_gtfs
    },
    "validation": {
        "poi": validate_poi
    }
}


def check_input(actions: list[str], datasets: list[str]) -> bool:
    """Check if input is valid.

    Args:
        actions (list[str]): Actions to perform.
        datasets (list[str]): Datasets to perform actions on.

    Raises:
        typer.Abort: If action is not supported.

    Returns:
        bool: True if input is valid.
    """
    # Check if action in action_dict keys
    for action in actions:
        if action not in action_dict.keys():
            typer.echo(f"Action {action} is not supported.")
            raise typer.Abort()

    # Check if dataset supports action if not print that dataset does not support action but continue
    for action in actions:
        for dataset in datasets:
            if dataset not in action_dict[action].keys():
                typer.echo(f"Dataset {dataset} does not support action {action}.")

    return True


def check_config_file_exists(data_set: str, region: str) -> bool:
    """Check if the configuration file exists."""
    config_path = os.path.join(
        settings.CONFIG_DIR,
        "data_variables",
        data_set,
        data_set + "_" + region + ".yaml",
    )
    if not os.path.isfile(config_path):
        typer.echo(f"Configuration file {config_path} does not exist.")
        raise typer.Abort()
    return True


@app.command()
def run(
    actions: str = typer.Option(None, "--actions", "-a"),
    region: str = typer.Option(None, "--region", "-r"),
    data_sets: str = typer.Option(None, "--datasets", "-d"),
):
    """Orchestrate the data preparation process."""
    all_actions = actions.split(",")
    data_sets = data_sets.split(",")

    # Check if all data sets are valid
    check_input(actions=all_actions, datasets=data_sets)

    # Loop through actions dicts and check if action and dataset are requested. If so, compute
    for action in all_actions:
        for dataset in data_sets:
            if dataset in action_dict[action].keys() and action in action_dict.keys():
                print_hashtags()
                if region is not None:
                    print_info(f"Performing {action} on {dataset} for region <{region}>")
                else:
                    print_info(f"Performing {action} on {dataset}")
                print_hashtags()

                if region is not None:
                    check_config_file_exists(data_set=dataset, region=region)
                    action_dict[action][dataset](region=region)
                else:
                    action_dict[action][dataset]()

if __name__ == "__main__":
    app()
