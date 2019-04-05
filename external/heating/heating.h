#ifndef __HEATING__
#define __HEATING__

#include "common.h" //Use here ONLY the things required for defining the struct (i.e. common.h for the ErrorMsg)

/**
 * All heating parameters and evolution that other modules need to know.
 */
enum f_eff_approx {f_eff_on_the_spot, f_eff_from_file};
enum chi_approx {chi_full_heating, chi_from_SSCK, chi_from_x_file, chi_from_z_file};

struct heating{

  /** @name - input parameters initialized by user in input module (all other quantities are computed in this module,
   *   given these parameters and the content of the 'precision', 'background', 'thermodynamics' and
   *  'primordial' structures) */

  //@{

  /* Exotic energy injection parameters */
  double annihilation_efficiency;/**< parameter describing CDM annihilation (f <sigma*v> / m_cdm, see e.g. 0905.0003) */
  double annihilation_cross_section;
  double DM_mass;
  double annihilation_variation;
  double annihilation_z;
  double annihilation_zmax;
  double annihilation_zmin;
  double annihilation_f_halo;
  double annihilation_z_halo;

  double decay_efficiency;       /**< parameter describing CDM decay (f/tau, see e.g. 1109.6322)*/

  /* Injection efficiency */
  int f_eff_type;
  char *f_eff_file;

  /* Deposition function and injection efficiency */
  int chi_type;
  char *chi_z_file;
  char *chi_x_file;

  /* Approximation for energy injection of acoustic waves dissipation */
  int heating_rate_acoustic_diss_approx;

  //@}


  /** @name - Imported parameters */

  //@{

  /* Parameters from background structure */
  /* Redshift independent, i.e. defined in heating_init */
  double H0;
  double T_g0;
  double Omega0_b;
  double rho0_cdm;
  /* Redshift dependent, i.e. defined in heating_at_z or heating_at_z_second_order */
  double H;
  double a;
  double R;
  double rho_g;
  double rho_cdm;
  double rho_dcdm;
  double T_b;
  double x_e;

  /* Parameters from thermodynamics structure */
  /* Redshift independent, i.e. defined in heating_init */
  double Y_He;
  double f_He;
  double fHe;
  double heat_capacity;

  double N_e0;
  double nH;
  /* Redshift dependent, i.e. defined in heating_at_z or heating_at_z_second_order */
  double dkappa;
  double dkD_dz;
  double kD;

  /* Parameters from primordial structure */
  double k_max;
  double k_min;
  double k_size;
  double* k;
  double* pk_primordial_k;

  //@}


  /** @name - Public tables and parameters */

  //@{

  /* Redshift tables */
  double* z_table;
  int z_size;

  double tol_z_table;
  int filled_until_index_z;
  double filled_until_z;

  int last_index_z_feff;
  int last_index_z_chi;
  int last_index_z_inj;
  int last_index_z;

  int index_z_store;

  /* TODO */
  int last_index_chix;

  /* Energy injection table */
  double** injection_table;
  int index_inj_cool;
  int index_inj_diss;
  int index_inj_DM_ann;
  int index_inj_DM_dec;
  int index_inj_tot;
  //int index_dep_lowE;
  int inj_size;                  /** All contributions + total */

  /* Deposition function tables */
  double* chiz_table;
  int chiz_size;
  double* chix_table;
  int chix_size;

  double** deposition_table; /* The table of energy depositions into the IGM of different deposition types */
  double* photon_dep_table;  /* The table of energy depositions into the photon fluid */
  double* chi;
  int index_dep_heat;
  int index_dep_ionH;
  int index_dep_ionHe;
  int index_dep_lya;
  int index_dep_lowE;
  int dep_size;

  /* Energy deposition vector */
  double* pvecdeposition;

  /* Injection efficiency table */
  double f_eff;
  int feff_z_size;
  double* feff_table;

   //@}

  /** @name - Flags and technical parameters */

  //@{

  /* Flags */
  int has_exotic_injection;

  int has_DM_ann;
  int has_DM_dec;

  int to_store;

  /* Book-keeping */
  int heating_verbose;

  ErrorMsg error_message;

};



/**************************************************************/

/* *
 * Putting this down here is important, because of the special nature of this module.
 * This allows the struct heating to already be defined and thus be a normal member
 * (as opposed to a pointer member) of the struct thermo in thermodynamics.h
 * */
struct background;
struct thermo;
struct perturbs;
struct primordial;

/* @cond INCLUDE_WITH_DOXYGEN */
/*
 * Boilerplate for C++
 */
#ifdef __cplusplus
extern "C" {
#endif

  /* Allocate, define indeces for and free heating tables */
  int heating_init(struct precision * ppr,
                   struct background* pba,
                   struct thermo* pth);

  int heating_indices(struct thermo* pth);

  int heating_free(struct thermo* pth);

  /* Main functions */
  int heating_calculate_at_z(struct background* pba,
                             struct thermo* pth,
                             double x,
                             double z,
                             double Tmat,
                             double* pvecback);

  int heating_energy_injection_at_z(struct heating* phe,
                                    double z,
                                    double* dEdz_inj);

  int heating_deposition_function_at_z(struct heating* phe,
                                       double x,
                                       double z);

  int heating_photon_at_z(struct thermo* pth,
                          double z,
                          double* heat);

  int heating_baryon_at_z(struct thermo* pth,
                          double z);

  int heating_add_noninjected(struct background* pba,
                              struct thermo* pth,
                              struct perturbs* ppt,
                              struct primordial* ppm);

  /* Branching ratios into the different channels */
  int heating_read_chi_z_from_file(struct precision* ppr,
                                   struct heating* phe);

  int heating_read_chi_x_from_file(struct precision* ppr,
                                   struct heating* phe);

  /* Efficiency of energy deposition */
  int heating_read_feff_from_file(struct precision* ppr,
                                  struct heating* phe);

  /* Heating functions */
  int heating_rate_adiabatic_cooling(struct heating * phe,
                                     double z,
                                     double * energy_rate);

  int heating_rate_acoustic_diss(struct heating * phe,
                                 double z,
                                 double * energy_rate);

  int heating_rate_DM_annihilation(struct heating * phe,
                                   double z,
                                   double * energy_rate);

  int heating_rate_DM_decay(struct heating * phe,
                            double z,
                            double * energy_rate);

#ifdef __cplusplus
}
#endif

#endif
