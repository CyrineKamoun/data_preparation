import os
import subprocess

import yaml

from src.config.osm_dict import OSM_germany, OSM_tags
from src.core.config import settings
from src.utils.utils import download_link, print_info


class Config:
    """Reads the config file and returns the config variables.
    """
    def __init__(self, name: str, region: str):
        #TODO: Add validation of config files here

        self.dataset_dir = os.path.join(settings.INPUT_DATA_DIR, name)

        # Read the base config file
        self.config_base = self.read_config("config.yaml")

        # Read the specific config file for the dataset
        config_file_name = f"{name}_{region}.yaml"
        self.config = self.read_config(os.path.join("data_variables", name, config_file_name))

        if name != "global":
            self.name = name
            self.collection = self.config.get("collection")
            self.preparation = self.config.get("preparation")
            self.export = self.config.get("export")
            self.subscription = self.config.get("subscription")
            self.analysis = self.config.get("analysis")
            self.pbf_data = self.config.get("region_pbf")

        if region == 'europe' and name == 'poi':
            # get the Geofabrik download links that are not in other config files
            folder_path = os.path.join(settings.CONFIG_DIR, "data_variables", name)
            self.regions = list({file.split("_")[1].split(".")[0] for file in os.listdir(folder_path) if file.endswith(".yaml")})
            for file in os.listdir(folder_path):
                if file.endswith(".yaml") and file != f"{name}_{region}.yaml" and file != f"{name}_{region}_all.yaml":
                    other_config = self.read_config(os.path.join("data_variables", name, file))
                    self.pbf_data = [item for item in self.pbf_data if item not in other_config.get("region_pbf", [])]
        else:
            self.regions = [region]

    def read_config(self, config_file_path):
        """Reads a YAML config file and returns the configuration."""
        config_path = os.path.join(settings.CONFIG_DIR, config_file_path)
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return config

    def osm2pgsql_create_style(self):
        add_columns = self.collection["additional_columns"]
        osm_tags = self.collection["osm_tags"]

        f = open(
            os.path.join(settings.CONFIG_DIR, "style_template.style"), "r"
        )
        sep = "#######################CUSTOM###########################"
        text = f.read()
        text = text.split(sep, 1)[0]

        f1 = open(
            os.path.join(
                self.dataset_dir, "osm2pgsql.style"
            ),
            "w",
        )
        f1.write(text)
        f1.write(sep)
        f1.write("\n")

        print_info(f"Creating osm2pgsql for {self.name}...")

        for column in add_columns:
            if column in ["railway", "highway"]:
                style_line = f"node,way  {column}  text  polygon"
                f1.write(style_line)
                f1.write("\n")
            else:
                style_line = f"node,way  {column}  text  linear"
                f1.write(style_line)
                f1.write("\n")

        if osm_tags is None:
            keys = ["aerialway", "aeroway", "amenity", "barrier", "boundary", "building", "craft", "emergency", "geological", "healthcare", "historic", "leisure", "man_made", "military", "nature", "office", "power", "public_transport", "railway", "shop", "sport", "tourism", "water"]
            for key in keys:
                f1.write(f"node,way  {key}  text  linear\n")
        else:
            for tag in osm_tags:
                if tag in ["railway", "highway"]:
                    style_line = f"node,way  {tag}  text  linear"
                    f1.write(style_line)
                    f1.write("\n")
                else:
                    style_line = f"node,way  {tag}  text  polygon"
                    f1.write(style_line)
                    f1.write("\n")

    def download_db_schema(self):
        """Download database schema from PostGIS database."""
        download_link(
            settings.INPUT_DATA_DIR, self.config_base["db_schema"], "dump.tar"
        )


