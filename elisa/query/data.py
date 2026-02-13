import pandas as pd
import numpy as np
import os
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

from typing import Optional, Dict, Union, List, Tuple

from pathlib import Path
import logging

#import warnings
#warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Default columns for Gaia queries
DEFAULT_COLUMNS = """source_id, ra, ra_error, dec, dec_error, l, b, parallax, parallax_error,
                    pmra, pmra_error, pmdec, pmdec_error, 
                    phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, 
                    phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error,
                    phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, 
                    radial_velocity, radial_velocity_error"""


VIZIER_MIRRORS = {
    'cds': 'https://vizier.cds.unistra.fr',
    'harvard': 'https://vizier.cfa.harvard.edu',
    'tokyo': 'https://vizier.nao.ac.jp',
    'beijing': 'https://vizier.china-vo.org',
}


class ElisaQuery:

    def __init__(self, verbose: bool = True, vizier_server: str = 'harvard'):

        """
        Initialize ElisaQuery instance.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print status messages. If False, only logs to logger.
        vizier_server : str, default='harvard'
            VizieR mirror to use. Options: 'cds' (France), 'harvard' (USA),
            'tokyo' (Japan), 'beijing' (China), or a custom URL.
        """

        self.gaia_login = False
        self.df_cluster = pd.DataFrame()
        self.df_members = pd.DataFrame()
        self.verbose = verbose

        # Set VizieR server
        if vizier_server in VIZIER_MIRRORS:
            Vizier.VIZIER_SERVER = VIZIER_MIRRORS[vizier_server]
        else:
            Vizier.VIZIER_SERVER = vizier_server
        self._log(f"Using VizieR server: {Vizier.VIZIER_SERVER}")


    def _log(self, message: str, level: str = 'info') -> None:
        """Internal logging helper that respects verbose setting."""
        log_func = getattr(logger, level)
        log_func(message)
        if self.verbose:
            print(message)


    def login(self, path_to_credentials_file: Optional[str] = None,
                    user_credentials: Optional[Dict[str, str]] = None) -> None:
        """
        Log in to Gaia archive.
        
        Parameters
        ----------
        path_to_credentials_file : str, optional
            Path to credentials file
        user_credentials : dict, optional
            Dictionary with 'username' and 'password' keys
        """

        if path_to_credentials_file and user_credentials:
            raise ValueError("Provide either 'path_to_credentials_file' or 'user_credentials', not both.")

        if path_to_credentials_file:
            credentials_path = Path(path_to_credentials_file)
            if not credentials_path.exists():
                raise FileNotFoundError(f"Credentials file does not exist: {credentials_path}")

            self._log("Logging into Gaia using credentials file.")
            Gaia.login(credentials_file=str(credentials_path))

        elif user_credentials:
            try:
                username = user_credentials["username"]
                password = user_credentials["password"]
            except KeyError as exc:
                raise ValueError("user_credentials must contain 'username' and 'password'.") from exc

            self._log("Logging into Gaia using user credentials dictionary.")
            Gaia.login(user=username, password=password)

        else:
            raise ValueError("No authentication method provided. Specify either a credentials file or a user credentials dictionary.")

        self.gaia_login = True
        self._log("Successfully logged into Gaia archive.")


    def logout(self) -> None:
        """Log out from Gaia archive."""
        if not self.gaia_login:
            self._log("Not logged in.", level='warning')
            return

        Gaia.logout()
        self.gaia_login = False
        self._log("Successfully logged out from Gaia archive.")


    def load_gaia_tables(self, include_shared: bool = True) -> List[str]:
        """
        Load and display available Gaia tables.

        Parameters
        ----------
        include_shared : bool, default=True
            Whether to include shared tables (requires login)

        Returns
        -------
        list of str
            List of available table names
        """
        if include_shared and not self.gaia_login:
            self._log("Warning: include_shared=True but not logged in. "
                     "Shared tables will not be included.", level='warning')
            include_shared = False

        tables = Gaia.load_tables(only_names=True, include_shared_tables=include_shared)

        table_names = [table.get_qualified_name() for table in tables]

        self._log(f"Found {len(table_names)} available Gaia tables.")
        if self.verbose:
            for name in table_names:
                print(f"  {name}")

        return table_names


    def load_catalog(self, catalog_name: str = 'alfonso-2024', 
                    load_members: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load star cluster catalogs from VizieR.
        
        Parameters
        ----------
        catalog_name : str
            Catalog identifier. Options: 'alfonso-2024', 'cantat-gaudin-2020', 
            'hunt-2023', 'vasiliev-2021'
        load_members : bool, default=False
            Whether to load cluster members
        
        Returns
        -------
        tuple of pd.DataFrame
            (clusters_df, members_df)
        """
        
        _print_str = 'clusters and members' if load_members else 'clusters'

        catalog_info = {
            'cantat-gaudin-2020': ('Cantat-Gaudin et al. 2020', 'J/A+A/640/A1/table1', 'J/A+A/640/A1/nodup'),
            'alfonso-2024': ('Alfonso et al. 2024', 'J/A+A/689/A18/clusters', 'J/A+A/689/A18/members'),
            'hunt-2023': ('Hunt et al. 2023', 'J/A+A/673/A114/clusters', 'J/A+A/673/A114/members'),
            'vasiliev-2021': ('Vasiliev et al. 2021', 'J/MNRAS/505/5978/tablea1', None),
        }

        if catalog_name not in catalog_info:
            available = ', '.join(catalog_info.keys())
            raise ValueError(f"Catalog '{catalog_name}' is not recognized. Available: {available}")

        author, catalog, catalog_members = catalog_info[catalog_name]
        self._log(f"Loading {_print_str} from {author}")

        try:
            query = Vizier(catalog=catalog, row_limit=-1).query_constraints()[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load catalog '{catalog_name}': \n{str(e)}")
        
        self.df_cluster = query.to_pandas()
        self.df_cluster = self.df_cluster.rename(columns={col: 'cluster' for col in ['Name', 'Cluster'] if col in self.df_cluster.columns})
        self.df_cluster['cluster'] = self.df_cluster['cluster'].str.replace(' ', '_')

        self._log(f"Loaded {len(self.df_cluster)} clusters from the {catalog_name} catalog.")

        if load_members:
            if catalog_members is None:
                raise ValueError(f"Members for '{catalog_name}' are not available, please set load_members=False.")

            try:
                query_members = Vizier(catalog=catalog_members, row_limit=-1).query_constraints()[0]
            except Exception as e:
                raise RuntimeError(f"Failed to load members from '{catalog_name}': \n{str(e)}")

            self.df_members = query_members.to_pandas()
            self._log(f"Loaded {len(self.df_members)} members from the {catalog_name} catalog.")
            return self.df_cluster, self.df_members

        return self.df_cluster, pd.DataFrame()


    def gaia_query(self,
                coordinates: Dict[str, float],
                name_for_data: str = 'gaia_field',
                parallax_over_error: Optional[float] = 10,
                min_parallax: Optional[float] = None,
                max_parallax: Optional[float] = None,
                path_to_save_data: Optional[str] = None,
                columns: str = 'all',
                additional_filters: Optional[Union[str, List[str]]] = None,
                use_default_filters: bool = True) -> pd.DataFrame:
        """
        Query Gaia Data Release 3 catalog with spatial cone search and quality filters.

        Parameters
        ----------
        coordinates : dict
            Dictionary with 'ra', 'dec' (degrees), and 'radius' (degrees)
        name_for_data : str, default='gaia_field'
            Name for saved file (if path_to_save_data is provided)
        parallax_over_error : float, optional, default=10
            Minimum parallax/error ratio. Set to None to disable.
        min_parallax : float, optional
            Minimum parallax (mas). Can be used alone or with max_parallax.
        max_parallax : float, optional
            Maximum parallax (mas). Can be used alone or with min_parallax.
        path_to_save_data : str, optional
            Directory to save results. If None, results are not saved to file.
        columns : str, default='all'
            Columns to retrieve: 'all' for all columns, 'default' for common
            astrometric/photometric columns, or a custom column string.
        additional_filters : str or list of str, optional
            Additional SQL filter conditions to append.
        use_default_filters : bool, default=True
            Whether to apply default quality filters (proper motion, RUWE, etc.)

        Returns
        -------
        pd.DataFrame
            Query results
        """
        # Validate inputs before any side effects
        if coordinates is None:
            raise ValueError("coordinates dictionary with RA, Dec, and radius must be provided.")

        required_keys = ['ra', 'dec', 'radius']
        if not all(key in coordinates for key in required_keys):
            raise ValueError(f"coordinates must contain: {required_keys}")

        # Handle column selection
        if columns == 'all':
            columns = '*'
        elif columns == 'default':
            columns = DEFAULT_COLUMNS
        # else: use columns as provided

        filters = []

        if parallax_over_error is not None:
            filters.append(f"gaia.parallax IS NOT NULL AND gaia.parallax_over_error > {parallax_over_error}")

        # Flexible parallax range filtering
        if min_parallax is not None and max_parallax is not None:
            filters.append(f"gaia.parallax BETWEEN {min_parallax} AND {max_parallax}")
        elif min_parallax is not None:
            filters.append(f"gaia.parallax >= {min_parallax}")
        elif max_parallax is not None:
            filters.append(f"gaia.parallax <= {max_parallax}")

        if use_default_filters:
            default_filters = [
                "gaia.pmra IS NOT NULL",
                "gaia.pmdec IS NOT NULL",
                "gaia.ruwe < 1.4",
                "gaia.phot_g_mean_flux_over_error > 10",
                "gaia.phot_rp_mean_flux_over_error > 10",
                "gaia.phot_bp_mean_flux_over_error > 10",
                "gaia.astrometric_excess_noise < 1",
                "gaia.visibility_periods_used > 6",
                "gaia.phot_bp_rp_excess_factor < 1.3 + 0.060*POWER(gaia.phot_bp_mean_mag - gaia.phot_rp_mean_mag, 2)",
                "gaia.phot_bp_rp_excess_factor > 1.0 + 0.015*POWER(gaia.phot_bp_mean_mag - gaia.phot_rp_mean_mag, 2)"
            ]
            filters.extend(default_filters)
        
        if additional_filters:
            if isinstance(additional_filters, str):
                filters.append(additional_filters)
            elif isinstance(additional_filters, list):
                filters.extend(additional_filters)
            else:
                raise ValueError("additional_filters must be a string or list of strings")
        
        where_clause = " AND ".join(filters) if filters else "1=1"

        query = f"""
                SELECT {columns}
                FROM gaiadr3.gaia_source AS gaia
                WHERE CONTAINS(POINT('ICRS', gaia.ra, gaia.dec), CIRCLE('ICRS', {coordinates['ra']}, {coordinates['dec']}, {coordinates['radius']})) = 1
                AND {where_clause}
                """
        
        try:
            job = Gaia.launch_job_async(query, name=name_for_data)
            result = job.get_results()
            df_result = result.to_pandas()

            if path_to_save_data is not None:
                os.makedirs(path_to_save_data, exist_ok=True)
                output_path = os.path.join(path_to_save_data, f"{name_for_data}.csv")
                df_result.to_csv(output_path, index=False)
                self._log(f"Query returned {len(df_result)} rows. Saved to {output_path}")
            else:
                self._log(f"Query returned {len(df_result)} rows.")

            return df_result

        except Exception as e:
            self._log(f"Gaia query failed: {str(e)}", level='error')
            raise


    def gaia_source_id(self,
                    source_id: Union[int, List[int], np.ndarray],
                    columns: str = 'all',
                    name_for_data: str = 'gaia_source',
                    path_to_save_data: Optional[str] = None,
                    chunk_size: int = 5000
                    ) -> pd.DataFrame:
        """
        Query Gaia DR3 by source_id(s).

        Parameters
        ----------
        source_id : int or list of int or numpy array
            Single source_id or list of source_ids to query
        columns : str, default='all'
            Columns to retrieve: 'all' for all columns, 'default' for common
            astrometric/photometric columns, or a custom column string.
        name_for_data : str, default='gaia_source'
            Name for saved file (if path_to_save_data is provided)
        path_to_save_data : str, optional
            Directory to save results. If None, results are not saved to file.
        chunk_size : int, default=5000
            Maximum number of source_ids per query (for large lists)

        Returns
        -------
        pd.DataFrame
            Query results
        """
        if isinstance(source_id, (int, np.integer)):
            source_ids = [int(source_id)]
        elif isinstance(source_id, (list, tuple, np.ndarray)):
            source_ids = [int(sid) for sid in source_id]
            if not source_ids:
                raise ValueError("source_id list cannot be empty")
        else:
            raise ValueError("source_id must be an int, list of ints, or numpy array")

        # Handle column selection
        if columns == 'all':
            columns = '*'
        elif columns == 'default':
            columns = DEFAULT_COLUMNS
        # else: use columns as provided

        if len(source_ids) > chunk_size:
            self._log(f"Querying {len(source_ids)} source_ids in chunks of {chunk_size}...")
            return self._query_source_ids_chunked(
                                                    source_ids=source_ids,
                                                    columns=columns,
                                                    name_for_data=name_for_data,
                                                    path_to_save_data=path_to_save_data,
                                                    chunk_size=chunk_size
                                                )

        return self._query_source_ids_single(
                                                source_ids=source_ids,
                                                columns=columns,
                                                name_for_data=name_for_data,
                                                path_to_save_data=path_to_save_data
                                            )


    def _query_source_ids_single(self,
                                source_ids: List[int],
                                columns: str,
                                name_for_data: str,
                                path_to_save_data: Optional[str]) -> pd.DataFrame:
        """
        Execute a single query for source_ids.

        Parameters
        ----------
        source_ids : list of int
            List of source_ids to query
        columns : str
            Columns to retrieve
        name_for_data : str
            Name for saved file
        path_to_save_data : str, optional
            Directory to save results. If None, results are not saved.

        Returns
        -------
        pd.DataFrame
            Query results
        """
        
        if len(source_ids) == 1:
            source_id_clause = f"gaia.source_id = {source_ids[0]}"
        else:
            source_id_list = ','.join(map(str, source_ids))
            source_id_clause = f" gaia.source_id IN ({source_id_list}) "
        
        query = f"""
                SELECT {columns}
                FROM gaiadr3.gaia_source AS gaia
                WHERE {source_id_clause}
                """
        
        try:
            job = Gaia.launch_job_async(query, name=name_for_data)
            result = job.get_results()
            df_result = result.to_pandas()
            
            if len(df_result) < len(source_ids):
                found_ids = set(df_result['source_id'].values)
                missing_ids = set(source_ids) - found_ids
                self._log(f"Warning: {len(missing_ids)} source_id(s) not found in Gaia DR3", level='warning')
                if len(missing_ids) <= 10:
                    self._log(f"Missing IDs: {missing_ids}", level='warning')

            if path_to_save_data is not None:
                os.makedirs(path_to_save_data, exist_ok=True)
                output_path = os.path.join(path_to_save_data, f"{name_for_data}.csv")
                df_result.to_csv(output_path, index=False)
                self._log(f"Query returned {len(df_result)} rows. Saved to {output_path}")
            else:
                self._log(f"Query returned {len(df_result)} rows.")

            return df_result

        except Exception as e:
            self._log(f"Gaia query failed: {str(e)}", level='error')
            raise


    def _query_source_ids_chunked(self,
                                source_ids: List[int],
                                columns: str,
                                name_for_data: str,
                                path_to_save_data: Optional[str],
                                chunk_size: int) -> pd.DataFrame:
        """
        Execute chunked queries for large lists of source_ids.

        Parameters
        ----------
        source_ids : list of int
            List of source_ids to query
        columns : str
            Columns to retrieve
        name_for_data : str
            Name for saved file
        path_to_save_data : str, optional
            Directory to save results. If None, results are not saved.
        chunk_size : int
            Maximum number of source_ids per query

        Returns
        -------
        pd.DataFrame
            Combined query results
        """
        
        chunks = [source_ids[i:i+chunk_size] for i in range(0, len(source_ids), chunk_size)]
        num_chunks = len(chunks)

        self._log(f"Splitting into {num_chunks} chunks...")

        dfs = []
        for i, chunk in enumerate(chunks, 1):
            self._log(f"Querying chunk {i}/{num_chunks} ({len(chunk)} source_ids)...")

            try:
                df_chunk = self._query_source_ids_single(
                    source_ids=chunk,
                    columns=columns,
                    name_for_data=f"{name_for_data}_chunk{i}",
                    path_to_save_data=path_to_save_data,
                    save_to_file=False  # Don't save chunks individually
                )
                dfs.append(df_chunk)

            except Exception as e:
                self._log(f"Warning: Chunk {i} failed: {str(e)}", level='warning')
                continue

        if not dfs:
            raise RuntimeError("All chunks failed to query. No data retrieved.")

        df_result = pd.concat(dfs, ignore_index=True)

        if len(df_result) < len(source_ids):
            found_ids = set(df_result['source_id'].values)
            missing_ids = set(source_ids) - found_ids
            self._log(f"Warning: {len(missing_ids)} source_id(s) not found in Gaia DR3", level='warning')

        if path_to_save_data is not None:
            os.makedirs(path_to_save_data, exist_ok=True)
            output_path = os.path.join(path_to_save_data, f"{name_for_data}.csv")
            df_result.to_csv(output_path, index=False)
            self._log(f"Combined query returned {len(df_result)} rows. Saved to {output_path}")
        else:
            self._log(f"Combined query returned {len(df_result)} rows.")

        return df_result