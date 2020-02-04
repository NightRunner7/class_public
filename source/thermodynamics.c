/** @file thermodynamics.c Documented thermodynamics module
 *
 * Initially written by:
 * Julien Lesgourgues, 6.09.2010
 *
 * Restructured structs by:
 * Nils Schoeneberg, 27.02.2019
 *
 * Evolver implementation by:
 * Daniel Meinert
 *
 * Deals with the thermodynamical evolution.
 * This module has two purposes:
 *
 * - at the beginning, to initialize the thermodynamics, i.e. to
 *   integrate the thermodynamical equations, and store all
 *   thermodynamical quantities as a function of redshift inside an
 *   interpolation table. The current version of recombination is
 *   based on RECFAST v1.5. The current version of reionization is
 *   based on exactly the same reionization function as in CAMB, in
 *   order to make allow for comparison. It should be easy to
 *   generalize the module to more complicated reionization histories.
 *
 * - to provide a routine which allow other modules to evaluate any
 *   thermodynamical quantities at a given redshift value (by
 *   interpolating within the interpolation table).
 *
 *
 * The logic is the following:
 *
 * - If RECFAST or HYREC are chosen, we use their respective
 *   differential equations or evolvers to compute the free electron
 *   fraction x at each step requested by the z output array
 *   In the RECFAST case, we additionally evolve the x_H and x_He
 *   free hydrogen and helium fractions
 *   In the HYREC case, only the temperature of baryons Tmat is evolved,
 *   the rest is done internally in the wrapper of hyrec
 *
 * - small detail: one of the columns contains the maximum variation
 *   rate of a few relevant thermodynamical quantities. This rate
 *   will be used for defining automatically the sampling step size in
 *   the perturbation module. Hence, the exact value of this rate is
 *   unimportant, but its order of magnitude at a given z defines the
 *   sampling precision of the perturbation module. Hence, it is
 *   harmless to use a smoothing routine in order to make this rate
 *   look nicer, although this will not affect the final result
 *   significantly. The last step in the thermodynamics_init module is
 *   to perform this smoothing.
 *
 * In summary, the following functions can be called from other modules:
 *
 * -# thermodynamics_init at the beginning (but after background_init)
 * -# thermodynamics_at_z at any later time
 * -# thermodynamics_free at the end, when no more calls to thermodynamics_at_z are needed
 */

#include "thermodynamics.h"

#ifdef HYREC
#include "history.h"
#ifndef TWOG_FILE
#include "hyrectools.h"
#include "helium.h"
#include "hydrogen.h"
#include "hyrec_params.h"
#endif
#endif

/**
 * Thermodynamics quantities at given redshift z.
 * Evaluates all thermodynamics quantities at a given value of the redshift by reading the pre-computed table and interpolating.
 *
 * @param pba          Input: pointer to background structure
 * @param pth          Input: pointer to the thermodynamics structure (containing pre-computed table)
 * @param z            Input: redshift
 * @param inter_mode   Input: interpolation mode (normal or growing_closeby)
 * @param last_index   Input/Output: index of the previous/current point in the interpolation array (input only for closeby mode, output for both)
 * @param pvecback     Input: vector of background quantities (used only in case z>z_initial for getting ddkappa and dddkappa; in that case,
                            should be already allocated and filled, with format short_info or larger; in other cases, will be ignored)
 * @param pvecthermo Output: vector of thermodynamics quantities (assumed to be already allocated)
 * @return the error status
 */

int thermodynamics_at_z(
                        struct background * pba,
                        struct thermo * pth,
                        double z,
                        short inter_mode,
                        int * last_index,
                        double * pvecback,
                        double * pvecthermo
                        ) {

  /** Summary: */

  /** - define local variables */
  double x0;
  double Vrms_idm_b2, T_diff_idm_b, Tb_in_eV, m_b, fHe; //DCH

  /* The fact that z is in the pre-computed range 0 <= z <= z_initial will be checked in the interpolation routines below. Before
     trying to interpolate, allow the routine to deal with the case z > z_intial: then, all relevant quantities can be extrapolated
     using simple analytic approximations */

  if (z >= pth->z_table[pth->tt_size-1]) {

    /* ionization fraction assumed to remain constant at large z */
    x0= pth->thermodynamics_table[(pth->tt_size-1)*pth->th_size+pth->index_th_xe];
    pvecthermo[pth->index_th_xe] = x0;

    /* Calculate dkappa/dtau (dkappa/dtau = a n_e x_e sigma_T = a^{-2} n_e(today) x_e sigma_T in units of 1/Mpc) */
    pvecthermo[pth->index_th_dkappa] = (1.+z) * (1.+z) * pth->n_e * x0 * _sigma_ * _Mpc_over_m_;

    /* tau_d scales like (1+z)**2 */
    pvecthermo[pth->index_th_tau_d] = pth->thermodynamics_table[(pth->tt_size-1)*pth->th_size+pth->index_th_tau_d]*pow((1+z)/(1.+pth->z_table[pth->tt_size-1]),2);

    if (pth->compute_damping_scale == _TRUE_) {

      /* r_d scales like (1+z)**-3/2 */
      pvecthermo[pth->index_th_r_d] = pth->thermodynamics_table[(pth->tt_size-1)*pth->th_size+pth->index_th_r_d]*pow((1+z)/(1.+pth->z_table[pth->tt_size-1]),-1.5);

    }

    /* Calculate d2kappa/dtau2 = dz/dtau d/dz[dkappa/dtau] given that [dkappa/dtau] proportional to (1+z)^2 and dz/dtau = -H */
    pvecthermo[pth->index_th_ddkappa] = -pvecback[pba->index_bg_H] * 2. / (1.+z) * pvecthermo[pth->index_th_dkappa];

    /* Calculate d3kappa/dtau3 given that [dkappa/dtau] proportional to (1+z)^2 */
    pvecthermo[pth->index_th_dddkappa] = (pvecback[pba->index_bg_H]*pvecback[pba->index_bg_H]/ (1.+z) - pvecback[pba->index_bg_H_prime]) * 2. / (1.+z) * pvecthermo[pth->index_th_dkappa];

    /* \f$ exp^{-\kappa}, g, g', g'' \f$ can be set to zero: they are used only for computing the source functions in the
       perturbation module; but source functions only need to be sampled below z_initial (the condition that
       z_start_sources<z_initial is checked in the perturbation module) */
    pvecthermo[pth->index_th_exp_m_kappa] = 0.;
    pvecthermo[pth->index_th_g]=0.;
    pvecthermo[pth->index_th_dg]=0.;
    pvecthermo[pth->index_th_ddg]=0.;

    /* Calculate Tb */
    pvecthermo[pth->index_th_Tb] = pba->T_cmb*(1.+z);

    /* Tb derivative, needed for IDM_B DCH */
    pvecthermo[pth->index_th_dTb] = pba->T_cmb;

    /* Calculate cb2 (cb2 = (k_B/mu) Tb (1-1/3 dlnTb/dlna) = (k_B/mu) Tb (1+1/3 (1+z) dlnTb/dz)) */
    /* note that m_H / mu = 1 + (m_H/m_He-1) Y_p + x_e (1-Y_p) */
    pvecthermo[pth->index_th_cb2] = _k_B_ / ( _c_ * _c_ * _m_H_ ) * (1. + (1./_not4_ - 1.) * pth->YHe + x0 * (1.-pth->YHe)) * pba->T_cmb * (1.+z) * 4. / 3.;

    /* derivatives of baryon sound speed (only computed if some non-minimal tight-coupling schemes is requested) */
    if (pth->compute_cb2_derivatives == _TRUE_) {

      /* since cb2 proportional to (1+z) or 1/a, its derivative wrt conformal time is given by dcb2 = - a H cb2 */
      pvecthermo[pth->index_th_dcb2] = - pvecback[pba->index_bg_H] * pvecback[pba->index_bg_a] * pvecthermo[pth->index_th_cb2];

      /* then its second derivative is given by ddcb2 = - a H' cb2 */
      pvecthermo[pth->index_th_ddcb2] = - pvecback[pba->index_bg_H_prime] * pvecback[pba->index_bg_a] * pvecthermo[pth->index_th_cb2];
    }

    /* in this regime, variation rate = dkappa/dtau */
    pvecthermo[pth->index_th_rate] = pvecthermo[pth->index_th_dkappa];

    /* we set the initial properties for idm_b here DCH */
    if(pth->has_idm_b == _TRUE_){
      /* constants used in the scattering rate and temperatures */
      fHe = 1-pth->YHe;
      Tb_in_eV = pvecthermo[pth->index_th_Tb]*_k_B_/_eV_ ;
      m_b = _m_p_*_c_*_c_/_eV_;

      Vrms_idm_b2 = 1.e-8; //At early times, we no longer need the if statement, as z > 10^3

      if(pth->n_index_idm_b == -4){ //special treatment for this case, in which the baryons and DM are not tightly coupled at early times
        pvecthermo[pth->index_th_T_idm_b] = 0.;
        pvecthermo[pth->index_th_dT_idm_b] = 0.; //factors (1+z) cancel out
      }
      else {
        pvecthermo[pth->index_th_T_idm_b] = pba->T_cmb*(1.+z);
        pvecthermo[pth->index_th_dT_idm_b] = pba->T_cmb; //factors (1+z) cancel out
      }

      T_diff_idm_b = (Tb_in_eV/m_b)+(pvecthermo[pth->index_th_T_idm_b]*_k_B_/_eV_/pth->m_idm)+(Vrms_idm_b2/3.0); //T and m are all in eV
      pvecthermo[pth->index_th_c_idm_b2] = pvecthermo[pth->index_th_T_idm_b]/pth->m_idm*_eV_/(_c_*_c_);

      pvecthermo[pth->index_th_R_idm_b] = (pvecback[pba->index_bg_a]*pvecback[pba->index_bg_rho_b]*pth->cross_idm_b*pth->n_coeff_idm_b/(m_b+pth->m_idm))
        *powf(T_diff_idm_b,(pth->n_index_idm_b+1.0)/2.0)*fHe
        *(3.e-4*powf(_c_,4.)/(8.*_PI_*_Mpc_over_m_*_G_*_eV_)); //conversion coefficient for the units

      pvecthermo[pth->index_th_R_idm_b_prime] = (pvecback[pba->index_bg_rho_b]*pth->cross_idm_b*pth->n_coeff_idm_b*fHe/(m_b+pth->m_idm))
        *powf(T_diff_idm_b,((pth->n_index_idm_b-1.0)/2.0))
        *(-pvecback[pba->index_bg_a]*pvecback[pba->index_bg_a]*T_diff_idm_b
          +pvecback[pba->index_bg_a]*((pth->n_index_idm_b+1.0)/2.0)*(pvecthermo[pth->index_th_dTb]/m_b + pvecthermo[pth->index_th_dT_idm_b]/pth->m_idm))
        *(3.e-4*powf(_c_,4.)/(8.*_PI_*_Mpc_over_m_*_G_*_eV_)); //conversion coefficient for the units, might need some checking DCH
    }

  }

  /** - interpolate in table with array_interpolate_spline (normal mode) or array_interpolate_spline_growing_closeby (closeby mode) */

  else {

    /* some very specific cases require linear interpolation because of a break in the derivative of the functions */
    if (((pth->reio_parametrization == reio_half_tanh) && (z < 2*pth->z_reio))
        || ((pth->reio_parametrization == reio_inter) && (z < 50.))) {

      class_call(array_interpolate_linear(pth->z_table,
                                          pth->tt_size,
                                          pth->thermodynamics_table,
                                          pth->th_size,
                                          z,
                                          last_index,
                                          pvecthermo,
                                          pth->th_size,
                                          pth->error_message),
                 pth->error_message,
                 pth->error_message);
    }

    /* in the "normal" case, use spline interpolation */
    else {

      if (inter_mode == pth->inter_normal) {

        class_call(array_interpolate_spline(pth->z_table,
                                            pth->tt_size,
                                            pth->thermodynamics_table,
                                            pth->d2thermodynamics_dz2_table,
                                            pth->th_size,
                                            z,
                                            last_index,
                                            pvecthermo,
                                            pth->th_size,
                                            pth->error_message),
                   pth->error_message,
                   pth->error_message);
      }

      if (inter_mode == pth->inter_closeby) {

        class_call(array_interpolate_spline_growing_closeby(pth->z_table,
                                                            pth->tt_size,
                                                            pth->thermodynamics_table,
                                                            pth->d2thermodynamics_dz2_table,
                                                            pth->th_size,
                                                            z,
                                                            last_index,
                                                            pvecthermo,
                                                            pth->th_size,
                                                            pth->error_message),
                   pth->error_message,
                   pth->error_message);

      }
    }

  }

  return _SUCCESS_;
}

/**
 * Initialize the thermo structure, and in particular the thermodynamics interpolation table.
 *
 * @param ppr   Input: pointer to precision structure
 * @param pba   Input: pointer to background structure
 * @param pth   Input/Output: pointer to initialized thermo structure
 * @return the error status
 */
int thermodynamics_init(struct precision * ppr,
                        struct background * pba,
                        struct thermo * pth){

  /** Summary: */

  /** - define local variables */

  /* vector of background values for calling background_at_tau */
  double * pvecback;
  int index_tau;

  /* structures for storing temporarily information on recombination and reionization */
  struct thermo_workspace * ptw;

  pth->has_idm_b = pba->has_idm_b; // This is just a convenient shift of variables to wrap everything here in if_idm_b statements, without needing to declare pba each time DCH

  /** - initialize pointers, allocate background vector */
  class_alloc(pvecback,pba->bg_size*sizeof(double),pba->error_message);

  if (pth->thermodynamics_verbose > 0){
    printf("Computing thermodynamics\n");
  }

  /** - compute and check primordial Helium fraction  */

  if (pth->YHe == _YHE_BBN_) {
    class_call(thermodynamics_helium_from_bbn(ppr,pba,pth),
               pth->error_message,
               pth->error_message);
  }
  if (pth->thermodynamics_verbose > 0) {
    printf(" -> with Y_He = %.4f\n",pth->YHe);
  }
  /* Set this parameter of central importance to ionization of hydrogen/helium */
  pth->fHe = pth->YHe/(_not4_ *(1.-pth->YHe));

  /* DCH: in the case of interacting DM-b, we need to adapt the start of integration time,
     to be sure we capture the time at which the two species are still coupled.
     This is the condition we need later to set the initial DM temperature. */
  /* DCH TODO: why is this with Omega and not with a has_idm_n flag? */
  if(pba->Omega0_idm_b > 0.){
    class_call(input_obtain_idm_b_z_ini(ppr,pba,pth),
               pth->error_message,
               pth->error_message);
  }

  /** - test, whether all parameters are in the correct regime */
  class_call(thermodynamics_test_parameters(ppr,pba,pth),
             pth->error_message,
             pth->error_message);

  /** - allocate and assign all temporary structures and indices */
  class_alloc(ptw, sizeof(struct thermo_workspace), pth->error_message);
  class_call(thermodynamics_workspace_init(ppr,pba,pth,ptw),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_indices(pth,ptw),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_lists(ppr,pba,pth,ptw),
             pth->error_message,
             pth->error_message);

  /** - assign all heating related properties and indices (not temporary) */
  class_call(heating_init(ppr,pba,pth),
             (pth->he).error_message,
             pth->error_message);

  /** - solve recombination and reionization and store values of \f$ z, x_e, d \kappa / d \tau, T_b, c_b^2 \f$ with thermodynamics_solve */
  class_call(thermodynamics_solve(ppr,pba,pth,ptw,pvecback),
             pth->error_message,
             pth->error_message);

  /** - the differential equation system is now completely solved  */

  /** - fill missing columns (quantities not computed during the differential evolution but related) */
  class_call(thermodynamics_calculate_remaining_quantities(ppr,pba,pth,pvecback),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_print_output(pba,pth),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_workspace_free(pth,ptw),
             pth->error_message,
             pth->error_message);
  free(pvecback);

  return _SUCCESS_;

}


/**
 * Free all memory space allocated by thermodynamics_init.
 *
 * @param pth Input/Output: pointer to thermo structure (to be freed)
 * @return the error status
 */
int thermodynamics_free(struct thermo * pth){

  /* Free all heating-related functions */
  class_call(heating_free(pth),
             (pth->he).error_message,
             pth->error_message);

  /* Free thermodynamics-related functions */
  free(pth->z_table);
  free(pth->tau_table);
  free(pth->thermodynamics_table);
  free(pth->d2thermodynamics_dz2_table);

  return _SUCCESS_;
}


/**
 * Test the thermo structure parameters for bounds and critical values.
 * Tests BBN Y_He fraction, annihilation injection parameters, divisions by zero, differential equation solving
 *
 * @param ppr   Input: pointer to precision structure
 * @param pba   Input: pointer to background structure
 * @param pth   Input: pointer to initialized thermo structure
 * @return the error status
 */
int thermodynamics_test_parameters(struct precision * ppr,
                                   struct background* pba,
                                   struct thermo * pth){

  /** Summary: */

  /** - check BBN Y_He fracion */
  class_test((pth->YHe < _YHE_SMALL_)||(pth->YHe > _YHE_BIG_),
             pth->error_message,
             "Y_He=%g out of bounds (%g<Y_He<%g)",pth->YHe,_YHE_SMALL_,_YHE_BIG_);

  /** - tests in order to prevent divisions by zero */
  class_test(_not4_ == 0.,
             pth->error_message,
             "stop to avoid division by zero");
  class_test(pth->YHe == 1.,
             pth->error_message,
             "stop to avoid division by zero");

  /** - tests for the differential equation solving */
  class_test(-ppr->thermo_z_initial > ppr->recfast_z_He_3,
             pth->error_message,
             "increase zinitial, as it is after HeliumIII recombination starts.");

  return _SUCCESS_;

}


/**
 * Assign value to each relevant index in vectors of thermodynamical quantities, and the reionization parameters
 *
 * @param pth   Input/Output: pointer to thermo structure
 * @param ptw   Input/Output: pointer to thermo workspace
 * @return the error status
 */
int thermodynamics_indices(struct thermo * pth,
                           struct thermo_workspace * ptw){

  /** Summary: */

  /** - define local variables */
  struct thermo_reionization_parameters* ptrp = ptw->ptrp;
  /* a running index for the vector of thermodynamics quantities */
  int index;

  /** - initialization of all indices and flags in thermo structure */
  index = 0;

  /* Free electron fraction */
  class_define_index(pth->index_th_xe,_TRUE_,index,1);
  /* Optical depth and related quantities */
  class_define_index(pth->index_th_dkappa,_TRUE_,index,1);
  class_define_index(pth->index_th_ddkappa,_TRUE_,index,1);
  class_define_index(pth->index_th_dddkappa,_TRUE_,index,1);
  class_define_index(pth->index_th_exp_m_kappa,_TRUE_,index,1);
  /* Visibility function + derivatives */
  class_define_index(pth->index_th_g,_TRUE_,index,1);
  class_define_index(pth->index_th_dg,_TRUE_,index,1);
  class_define_index(pth->index_th_ddg,_TRUE_,index,1);
  /* Baryon quantities, Temperature, Sound Speed, Drag time end */
  class_define_index(pth->index_th_Tb,_TRUE_,index,1);
  class_define_index(pth->index_th_dTb,_TRUE_,index,1);
  class_define_index(pth->index_th_cb2,_TRUE_,index,1);
  class_define_index(pth->index_th_tau_d,_TRUE_,index,1);
  /* Derivatives of baryon sound speed (only computed if some non-minimal tight-coupling schemes is requested) */
  class_define_index(pth->index_th_dcb2,pth->compute_cb2_derivatives,index,1);
  class_define_index(pth->index_th_ddcb2,pth->compute_cb2_derivatives,index,1);
  /* IDM-b quantities DCH */
  if(pth->has_idm_b == _TRUE_){
    class_define_index(pth->index_th_T_idm_b,_TRUE_,index,1);
    class_define_index(pth->index_th_dT_idm_b,_TRUE_,index,1);
    class_define_index(pth->index_th_c_idm_b2,_TRUE_,index,1);
    class_define_index(pth->index_th_R_idm_b,_TRUE_,index,1);
    class_define_index(pth->index_th_R_idm_b_prime,_TRUE_,index,1);
  }

  /* Important quantity defining the stepsize in perturbations.c */
  class_define_index(pth->index_th_rate,_TRUE_,index,1);
  /* Damping scale */
  class_define_index(pth->index_th_r_d,pth->compute_damping_scale,index,1);

  /* end of thermodynamics indices */

  pth->th_size = index;

  /** - initialization of all indicies of parameters of reionization function */

  index=0;

  class_define_index(ptrp->index_reio_start,_TRUE_,index,1);
  index++;

  /* case where x_e(z) taken like in CAMB (other cases can be added) */
  if ((pth->reio_parametrization == reio_camb) || (pth->reio_parametrization == reio_half_tanh)) {

    class_define_index(ptrp->index_reio_redshift,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_exponent,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_width,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_xe_before,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_xe_after,_TRUE_,index,1);
    class_define_index(ptrp->index_helium_fullreio_fraction,_TRUE_,index,1);
    class_define_index(ptrp->index_helium_fullreio_redshift,_TRUE_,index,1);
    class_define_index(ptrp->index_helium_fullreio_width,_TRUE_,index,1);
  }

  /* case where x_e(z) is binned */
  if (pth->reio_parametrization == reio_bins_tanh) {

    /* the code will not only copy here the "bin centers" passed in input. It will add an initial and final value for (z,xe). So
       this array has a dimension bigger than the bin center array */

    ptrp->reio_num_z=pth->binned_reio_num+2; /* add two values: beginning and end of reio */

    class_define_index(ptrp->index_reio_first_z,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_first_xe,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_step_sharpness,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_xe_before,_TRUE_,index,1);
  }

  /* case where x_e(z) has many tanh jumps */
  if (pth->reio_parametrization == reio_many_tanh) { //TODO :: is this the same as above?!

    /* the code will not only copy here the "jump centers" passed in input. It will add an initial and final value for (z,xe). So
       this array has a dimension bigger than the jump center array */

    ptrp->reio_num_z=pth->many_tanh_num+2; /* add two values: beginning and end of reio */

    class_define_index(ptrp->index_reio_first_z,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_first_xe,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_step_sharpness,_TRUE_,index,1);
    class_define_index(ptrp->index_reio_xe_before,_TRUE_,index,1);
  }

  /* case where x_e(z) must be interpolated */
  if (pth->reio_parametrization == reio_inter) {

    ptrp->reio_num_z=pth->reio_inter_num;

    class_define_index(ptrp->index_reio_first_z,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_first_xe,_TRUE_,index,ptrp->reio_num_z);
    class_define_index(ptrp->index_reio_xe_before,_TRUE_,index,1);
  }

  ptrp->reio_num_params = index;

  /* flags for calling the interpolation routine */
  pth->inter_normal=0;
  pth->inter_closeby=1;

  return _SUCCESS_;

}


/**
 * Initialize the lists (of redshift, tau, etc.) of the thermodynamics struct
 *
 * @param ppr   Input: pointer to precision structure
 * @param pba   Input: pointer to background structure
 * @param pth   Input/Output: pointer to thermo structure
 * @param ptw   Input: pointer to thermo workspace
 * @return the error status
 */
int thermodynamics_lists(struct precision * ppr,
                         struct background* pba,
                         struct thermo* pth,
                         struct thermo_workspace* ptw){

  /** Summary: */

  /** Define local variables */
  int index_tau, index_z;
  double zinitial,zlinear;

  pth->tt_size = ptw->Nz_tot;

  /** - allocate tables*/
  class_alloc(pth->tau_table,pth->tt_size*sizeof(double),pth->error_message);
  class_alloc(pth->z_table,pth->tt_size*sizeof(double),pth->error_message);
  class_alloc(pth->thermodynamics_table,pth->th_size*pth->tt_size*sizeof(double),pth->error_message);
  class_alloc(pth->d2thermodynamics_dz2_table,pth->th_size*pth->tt_size*sizeof(double),pth->error_message);

  /** - define time sampling */
  /* Initial z, and the z at which we switch to linear sampling */
  zinitial = ppr->thermo_z_initial;
  zlinear  = ppr->thermo_z_linear;
  /* -> Between zinitial and reionization_z_start_max, we use the spacing of recombination sampling */
  for(index_z=0; index_z <ptw->Nz_reco_log; index_z++) {
    pth->z_table[(pth->tt_size-1) - index_z] = -(-exp((log(zinitial)-log(zlinear))*(double)(ptw->Nz_reco_log-1-index_z) / (double)(ptw->Nz_reco_log-1)+log(zlinear)));
  }
  /* -> Between zinitial and reionization_z_start_max, we use the spacing of recombination sampling */
  for(index_z=0; index_z <ptw->Nz_reco_lin; index_z++) {
    pth->z_table[(pth->tt_size-1)-(index_z+ptw->Nz_reco_log)] = -(-(zlinear-ppr->reionization_z_start_max) * (double)(ptw->Nz_reco_lin-1-index_z) / (double)(ptw->Nz_reco_lin) - ppr->reionization_z_start_max);
  }
  /* -> Between reionization_z_start_max and 0, we use the spacing of reionization sampling, leaving out the first point to not double-count it */
  for(index_z=0; index_z <ptw->Nz_reio; index_z++) {
    pth->z_table[(pth->tt_size-1)-(index_z+ptw->Nz_reco)] = -(-ppr->reionization_z_start_max * (double)(ptw->Nz_reio-1-index_z) / (double)(ptw->Nz_reio));
  }

  for (index_tau=0; index_tau < pth->tt_size; index_tau++) {
    class_call(background_tau_of_z(pba,
                                   pth->z_table[index_tau],
                                   pth->tau_table+index_tau),
               pba->error_message,
               pth->error_message);
  }

  /** - store initial value of conformal time in the structure */
  pth->tau_ini = pth->tau_table[pth->tt_size-1];

  return _SUCCESS_;

}


/**
 * Infer the primordial helium fraction from standard BBN, as a function of the baryon density and expansion rate during BBN.
 *
 * This module is simpler then the one used in arXiv:0712.2826 because it neglects the impact of a possible significant chemical
 * potentials for electron neutrinos. The full code with xi_nu_e could be introduced here later.
 *
 * @param ppr   Input: pointer to precision structure
 * @param pba   Input: pointer to background structure
 * @param pth   Input/Output: pointer to initialized thermo structure
 * @return the error status
 */
int thermodynamics_helium_from_bbn(struct precision * ppr,
                                   struct background * pba,
                                   struct thermo * pth){

  /** Summary: */

  /** Define local variables */
  FILE * fA;
  char line[_LINE_LENGTH_MAX_];
  char * left;

  int num_omegab=0;
  int num_deltaN=0;

  double * omegab=NULL;
  double * deltaN=NULL;
  double * YHe=NULL;
  double * ddYHe=NULL;
  double * YHe_at_deltaN=NULL;
  double * ddYHe_at_deltaN=NULL;

  int array_line=0;
  double DeltaNeff;
  double omega_b;
  int last_index;
  double Neff_bbn, z_bbn, tau_bbn, *pvecback;

  /** - Infer effective number of neutrinos at the time of BBN */
  class_alloc(pvecback,pba->bg_size*sizeof(double),pba->error_message);

  /** - 8.6173e-11 converts from Kelvin to MeV. We randomly choose 0.1 MeV to be the temperature of BBN */
  z_bbn = 0.1/(8.6173e-11*pba->T_cmb)-1.0;

  class_call(background_tau_of_z(pba,
                                 z_bbn,
                                 &tau_bbn),
             pba->error_message,
             pth->error_message);

  class_call(background_at_tau(pba,
                               tau_bbn,
                               pba->long_info,
                               pba->inter_normal,
                               &last_index,
                               pvecback),
             pba->error_message,
             pth->error_message);

  Neff_bbn = (pvecback[pba->index_bg_Omega_r]
	      *pvecback[pba->index_bg_rho_crit]
	      -pvecback[pba->index_bg_rho_g])
    /(7./8.*pow(4./11.,4./3.)*pvecback[pba->index_bg_rho_g]);

  free(pvecback);

  //  printf("Neff early = %g, Neff at bbn: %g\n",pba->Neff,Neff_bbn);

  /** - compute Delta N_eff as defined in bbn file, i.e. \f$ \Delta N_{eff}=0\f$ means \f$ N_{eff}=3.046\f$ */
  DeltaNeff = Neff_bbn - 3.046;

  /* the following file is assumed to contain (apart from comments and blank lines):
     - the two numbers (num_omegab, num_deltaN) = number of values of BBN free parameters
     - three columns (omegab, deltaN, YHe) where omegab = Omega0_b h^2 and deltaN = Neff-3.046 by definition
     - omegab and deltaN are assumed to be arranged as:
     omegab1 deltaN1 YHe
     omegab2 deltaN1 YHe
     .....
     omegab1 delatN2 YHe
     omegab2 deltaN2 YHe
     .....
  */

  class_open(fA,ppr->sBBN_file, "r",pth->error_message);

  /* go through each line */
  while (fgets(line,_LINE_LENGTH_MAX_-1,fA) != NULL){

    /* eliminate blank spaces at beginning of line */
    left=line;
    while (left[0]==' '){
      left++;
    }

    /* check that the line is neither blank neither a comment. In ASCII, left[0]>39 means that first non-blank character might
       be the beginning of some data (it is not a newline, a #, a %, etc.) */
    if (left[0] > 39){

      /* if the line contains data, we must interpret it. If (num_omegab, num_deltaN)=(0,0), the current line must contain
         their values. Otherwise, it must contain (omegab, delatN, YHe). */
      if ((num_omegab==0) && (num_deltaN==0)) {

        /* read (num_omegab, num_deltaN), infer size of arrays and allocate them */
        class_test(sscanf(line,"%d %d",&num_omegab,&num_deltaN) != 2,
                   pth->error_message,
                   "could not read value of parameters (num_omegab,num_deltaN) in file %s\n",ppr->sBBN_file);

        class_alloc(omegab,num_omegab*sizeof(double),pth->error_message);
        class_alloc(deltaN,num_deltaN*sizeof(double),pth->error_message);
        class_alloc(YHe,num_omegab*num_deltaN*sizeof(double),pth->error_message);
        class_alloc(ddYHe,num_omegab*num_deltaN*sizeof(double),pth->error_message);
        class_alloc(YHe_at_deltaN,num_omegab*sizeof(double),pth->error_message);
        class_alloc(ddYHe_at_deltaN,num_omegab*sizeof(double),pth->error_message);
        array_line=0;

      }
      else{

        /* read (omegab, deltaN, YHe) */
        class_test(sscanf(line,"%lg %lg %lg",&(omegab[array_line%num_omegab]),
                                             &(deltaN[array_line/num_omegab]),
                                             &(YHe[array_line])
                          ) != 3,
                   pth->error_message,
                   "could not read value of parameters (omegab,deltaN,YHe) in file %s\n",ppr->sBBN_file);
        array_line ++;
      }
    }
  }

  fclose(fA);

  /** - spline in one dimension (along deltaN) */
  class_call(array_spline_table_lines(deltaN,
                                      num_deltaN,
                                      YHe,
                                      num_omegab,
                                      ddYHe,
                                      _SPLINE_NATURAL_,
                                      pth->error_message),
             pth->error_message,
             pth->error_message);

  omega_b=pba->Omega0_b*pba->h*pba->h;

  class_test(omega_b < omegab[0],
             pth->error_message,
             "You have asked for an unrealistic small value omega_b = %e. The corresponding value of the primordial helium fraction cannot be found in the interpolation table. If you really want this value, you should fix YHe to a given value rather than to BBN",
             omega_b);

  class_test(omega_b > omegab[num_omegab-1],
             pth->error_message,
             "You have asked for an unrealistic high value omega_b = %e. The corresponding value of the primordial helium fraction cannot be found in the interpolation table. If you really want this value, you should fix YHe to a given value rather than to BBN",
             omega_b);

  class_test(DeltaNeff < deltaN[0],
             pth->error_message,
             "You have asked for an unrealistic small value of Delta N_eff = %e. The corresponding value of the primordial helium fraction cannot be found in the interpolation table. If you really want this value, you should fix YHe to a given value rather than to BBN",
             DeltaNeff);

  class_test(DeltaNeff > deltaN[num_deltaN-1],
             pth->error_message,
             "You have asked for an unrealistic high value of Delta N_eff = %e. The corresponding value of the primordial helium fraction cannot be found in the interpolation table. If you really want this value, you should fix YHe to a given value rather than to BBN",
             DeltaNeff);

  /** - interpolate in one dimension (along deltaN) */
  class_call(array_interpolate_spline(deltaN,
                                      num_deltaN,
                                      YHe,
                                      ddYHe,
                                      num_omegab,
                                      DeltaNeff,
                                      &last_index,
                                      YHe_at_deltaN,
                                      num_omegab,
                                      pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - spline in remaining dimension (along omegab) */
  class_call(array_spline_table_lines(omegab,
                                      num_omegab,
                                      YHe_at_deltaN,
                                      1,
                                      ddYHe_at_deltaN,
                                      _SPLINE_NATURAL_,
                                      pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - interpolate in remaining dimension (along omegab) */
  class_call(array_interpolate_spline(omegab,
                                      num_omegab,
                                      YHe_at_deltaN,
                                      ddYHe_at_deltaN,
                                      1,
                                      omega_b,
                                      &last_index,
                                      &(pth->YHe),
                                      1,
                                      pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - deallocate arrays */
  free(omegab);
  free(deltaN);
  free(YHe);
  free(ddYHe);
  free(YHe_at_deltaN);
  free(ddYHe_at_deltaN);

  return _SUCCESS_;

}


/**
 * Calculate those thermodynamics quantities, which are not inside of the thermodynamics table already.
 * Additionally, any other requested quantities
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input/Output: pointer to initialized thermo structure
 * @param pvecback   Input: pointer to some allocated pvecback
 * @return the error status
 */
int thermodynamics_calculate_remaining_quantities(struct precision * ppr,
                                                  struct background * pba,
                                                  struct thermo* pth,
                                                  double* pvecback){

  /** Summary: */

  /** Define temporary variables */
  double tau_ini;
  double tau;
  int index_tau_max;
  int last_index_back = 0;

  /* index running over time*/
  int index_tau;

  /* same ordered in growing time rather than growing redshift */
  double * tau_table_growing;

  /* The temporary quantities stored in columns ddkappa and dddkappa will not be used anymore, so they can and WILL be overwritten by other
     intermediate steps of other computations */

  class_call(thermodynamics_calculate_conformal_drag_time(pba,pth,&last_index_back,pvecback),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_calculate_damping_scale(pba,pth,&last_index_back,pvecback),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_calculate_opticals(ppr,pth),
             pth->error_message,
             pth->error_message);

  /** - fill tables of second derivatives with respect to z (in view of spline interpolation) */
  class_call(array_spline_table_lines(pth->z_table,
                                      pth->tt_size,
                                      pth->thermodynamics_table,
                                      pth->th_size,
                                      pth->d2thermodynamics_dz2_table,
                                      _SPLINE_EST_DERIV_,
                                      pth->error_message),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_calculate_recombination_quantities(ppr,pba,pth,&last_index_back,pvecback),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_calculate_drag_quantities(ppr,pba,pth,&last_index_back,pvecback),
             pth->error_message,
             pth->error_message);

  return _SUCCESS_;

}


/**
 * Compute the baryon drag conformal time tau_d = [int_{tau_today}^{tau} dtau -dkappa_d/dtau]
 *
 * @param pba                Input: pointer to background structure
 * @param pth                Input/Output: pointer to initialized thermo structure
 * @param last_index_back    Input: temporary variable for storing index
 * @param pvecback           Input: Initialized vector of background quantities
 * @return the error status
 */
int thermodynamics_calculate_conformal_drag_time(struct background* pba,
                                                 struct thermo* pth,
                                                 int* last_index_back,
                                                 double* pvecback){

  /** Summary: */

  /** Define local variables */
  double R;
  int index_tau;

  /** - compute baryon drag interaction rate time minus one, -[1/R * kappa'], with R = 3 rho_b / 4 rho_gamma,
        stored temporarily in column ddkappa */
  *last_index_back = 0;

  for (index_tau=0; index_tau < pth->tt_size; index_tau++) {

    class_call(background_at_tau(pba,
                                 pth->tau_table[index_tau],
                                 pba->normal_info,
                                 pba->inter_closeby,
                                 last_index_back,
                                 pvecback),
               pba->error_message,
               pth->error_message);

    R = 3./4.*pvecback[pba->index_bg_rho_b]/pvecback[pba->index_bg_rho_g];

    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_ddkappa] =
      -1./R*pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_dkappa];

  }

  /** - compute second derivative of this rate, -[1/R * kappa']'', stored temporarily in column dddkappa */
  class_call(array_spline_table_line_to_line(pth->tau_table,
                                             pth->tt_size,
                                             pth->thermodynamics_table,
                                             pth->th_size,
                                             pth->index_th_ddkappa,
                                             pth->index_th_dddkappa,
                                             _SPLINE_EST_DERIV_,
                                             pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - compute tau_d = [int_{tau_today}^{tau} dtau -dkappa_d/dtau] */
  class_call(array_integrate_spline_table_line_to_line(pth->tau_table,
                                                       pth->tt_size,
                                                       pth->thermodynamics_table,
                                                       pth->th_size,
                                                       pth->index_th_ddkappa,
                                                       pth->index_th_dddkappa,
                                                       pth->index_th_tau_d,
                                                       pth->error_message),
             pth->error_message,
             pth->error_message);

  return _SUCCESS_;

}


/**
 * Compute the damping scale
 *   r_d = 2pi/k_d = 2pi * [int_{tau_ini}^{tau} dtau (1/kappa') 1/6 (R^2+16/15(1+R))/(1+R)^2]^1/2
 *                 =  2pi * [int_{tau_ini}^{tau} dtau (1/kappa') 1/6 (R^2/(1+R)+16/15)/(1+R)]^1/2
 *
 * which is like in CosmoTherm (CT), but slightly different from Wayne Hu (WH)'s thesis eq. (5.59):
 * The factor 16/15 in CT is 4/5 in WH, but 16/15 is taking more effects into account
 *
 * @param pba                Input: pointer to background structure
 * @param pth                Input/Output: pointer to initialized thermo structure
 * @param last_index_back    Input: temporary variable for storing index
 * @param pvecback           Input: Initialized vector of background quantities
 * @return the error status
 */
int thermodynamics_calculate_damping_scale(struct background* pba,
                                           struct thermo* pth,
                                           int* last_index_back,
                                           double* pvecback){

  /** Summary: */

  /** Define local variables */
  double R;
  /* Initial time and dkappa/dtau */
  double tau_ini,dkappa_ini;
  double* tau_table_growing;
  int index_tau;

  if (pth->compute_damping_scale == _TRUE_) {

    class_alloc(tau_table_growing,pth->tt_size*sizeof(double),pth->error_message);

    /* compute integrand and store temporarily in column "ddkappa" */
    for (index_tau=0; index_tau < pth->tt_size; index_tau++) {

      tau_table_growing[index_tau]=pth->tau_table[pth->tt_size-1-index_tau];

      class_call(background_at_tau(pba,
                                   tau_table_growing[index_tau],
                                   pba->normal_info,
                                   pba->inter_closeby,
                                   last_index_back,
                                   pvecback),
                 pba->error_message,
                 pth->error_message);

      R = 3./4.*pvecback[pba->index_bg_rho_b]/pvecback[pba->index_bg_rho_g];

      pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_ddkappa] =
        1./6./pth->thermodynamics_table[(pth->tt_size-1-index_tau)*pth->th_size+pth->index_th_dkappa]
        *(R*R/(1+R)+16./15.)/(1.+R);

    }

    /* compute second derivative of integrand, and store temporarily in column "dddkappa" */
    class_call(array_spline_table_line_to_line(tau_table_growing,
                                               pth->tt_size,
                                               pth->thermodynamics_table,
                                               pth->th_size,
                                               pth->index_th_ddkappa,
                                               pth->index_th_dddkappa,
                                               _SPLINE_EST_DERIV_,
                                               pth->error_message),
               pth->error_message,
               pth->error_message);

    /* compute integratal and store temporarily in column "g" */
     class_call(array_integrate_spline_table_line_to_line(tau_table_growing,
                                                          pth->tt_size,
                                                          pth->thermodynamics_table,
                                                          pth->th_size,
                                                          pth->index_th_ddkappa,
                                                          pth->index_th_dddkappa,
                                                          pth->index_th_g,
                                                          pth->error_message),
                pth->error_message,
                pth->error_message);

     free(tau_table_growing);

     /* we could now write the result as r_d = 2pi * sqrt(integral),
        but we will first better acount for the contribution frokm the tau_ini boundary.
        Close to this boundary, R=0 and the integrand is just 16/(15*6)/kappa'
        Using kappa' propto 1/a^2 and tau propro a during RD, we get the analytic result:
        int_0^{tau_ini} dtau / kappa' = tau_ini / 3 / kappa'_ini
        Thus r_d = 2pi * sqrt( 16/(15*6*3) * (tau_ini/ kappa'_ini) * integral) */

     tau_ini = pth->tau_table[pth->tt_size-1];
     dkappa_ini = pth->thermodynamics_table[(pth->tt_size-1)*pth->th_size+pth->index_th_dkappa];

     for (index_tau=0; index_tau < pth->tt_size; index_tau++) {

       pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_r_d] =
         2.*_PI_*sqrt(16./(15.*6.*3.)*tau_ini/dkappa_ini
                      +pth->thermodynamics_table[(pth->tt_size-1-index_tau)*pth->th_size+pth->index_th_g]);

     }

  }

  return _SUCCESS_;

}


/**
 * Calculate quantities relating to optical phenomena like kappa' and exp(-kappa) and the visibility function,
 * optical depth, etc.
 *
 * @param ppr   Input: pointer to precision structure
 * @param pth   Input/Output: pointer to thermo structure
 * @return the error status
 */
int thermodynamics_calculate_opticals(struct precision* ppr,
                                      struct thermo* pth){

  /** Summary: */

  /** Define local quantities */
  /* Visibility function value */
  double g;
  /* kappa derivative values*/
  double dkappa,ddkappa,dddkappa,expmkappa;
  int index_tau;

  /** - --> second derivative with respect to tau of dkappa (in view of spline interpolation) */
  class_call(array_spline_table_line_to_line(pth->tau_table,
                                             pth->tt_size,
                                             pth->thermodynamics_table,
                                             pth->th_size,
                                             pth->index_th_dkappa,
                                             pth->index_th_dddkappa,
                                             _SPLINE_EST_DERIV_,
                                             pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - --> first derivative with respect to tau of dkappa (using spline interpolation) */
  class_call(array_derive_spline_table_line_to_line(pth->tau_table,
                                                    pth->tt_size,
                                                    pth->thermodynamics_table,
                                                    pth->th_size,
                                                    pth->index_th_dkappa,
                                                    pth->index_th_dddkappa,
                                                    pth->index_th_ddkappa,
                                                    pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - --> compute -kappa = [int_{tau_today}^{tau} dtau dkappa/dtau], store temporarily in column "g" */
  class_call(array_integrate_spline_table_line_to_line(pth->tau_table,
                                                       pth->tt_size,
                                                       pth->thermodynamics_table,
                                                       pth->th_size,
                                                       pth->index_th_dkappa,
                                                       pth->index_th_dddkappa,
                                                       pth->index_th_g,
                                                       pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - --> compute visibility: \f$ g= (d \kappa/d \tau) e^{- \kappa} \f$ */

  /* loop on z (decreasing z, increasing time) */
  for (index_tau=pth->tt_size-1; index_tau>=0; index_tau--) {

    dkappa = pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_dkappa];
    ddkappa = pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_ddkappa];
    dddkappa = pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_dddkappa];
    expmkappa = exp(pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_g]);

    /** - ---> compute g */
    g = dkappa * expmkappa;

    /** - ---> compute exp(-kappa) */
    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_exp_m_kappa] = expmkappa;
    //printf("exp(-k)[%i] = %.10e \n",index_tau,expmkappa);

    /** - ---> compute g' (the plus sign of the second term is correct, see def of -kappa in thermodynamics module!) */
    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_dg] =
      (ddkappa + dkappa * dkappa) * expmkappa;

    /** - ---> compute g''  */
    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_ddg] =
      (dddkappa + dkappa * ddkappa * 3. + dkappa * dkappa * dkappa ) * expmkappa;

    /** - ---> store g */
    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_g] = g;

    /** - ---> compute variation rate */
    class_test(pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_dkappa] == 0.,
               pth->error_message,
               "variation rate diverges");

    pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_rate] =
      sqrt(pow(dkappa,2)+pow(ddkappa/dkappa,2)+fabs(dddkappa/dkappa));

  }

  /** - smooth the rate (details of smoothing unimportant: only the
      order of magnitude of the rate matters) */
  class_call(array_smooth(pth->thermodynamics_table,
                          pth->th_size,
                          pth->tt_size,
                          pth->index_th_rate,
                          ppr->thermo_rate_smoothing_radius,
                          pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - --> derivatives of baryon sound speed (only computed if some non-minimal tight-coupling schemes is requested) */
  if (pth->compute_cb2_derivatives == _TRUE_) {

    /** - ---> second derivative with respect to tau of cb2 */
    class_call(array_spline_table_line_to_line(pth->tau_table,
                                               pth->tt_size,
                                               pth->thermodynamics_table,
                                               pth->th_size,
                                               pth->index_th_cb2,
                                               pth->index_th_ddcb2,
                                               _SPLINE_EST_DERIV_,
                                               pth->error_message),
               pth->error_message,
               pth->error_message);

    /** - ---> first derivative with respect to tau of cb2 (using spline interpolation) */
    class_call(array_derive_spline_table_line_to_line(pth->tau_table,
                                                      pth->tt_size,
                                                      pth->thermodynamics_table,
                                                      pth->th_size,
                                                      pth->index_th_cb2,
                                                      pth->index_th_ddcb2,
                                                      pth->index_th_dcb2,
                                                      pth->error_message),
               pth->error_message,
               pth->error_message);
  }

  return _SUCCESS_;
}


/**
 * Calculate those quantities at the time of recombination,
 * Additionally compute time of free streaming and negligable visibility (tau_cut)
 *
 * @param ppr                Input: pointer to precision structure
 * @param pba                Input: pointer to background structure
 * @param pth                Input/Output: pointer to initialized thermo structure
 * @param last_index_back    Input: temporary variable for storing index
 * @param pvecback           Input: pointer to some allocated pvecback
 * @return the error status
 */
int thermodynamics_calculate_recombination_quantities(struct precision* ppr,
                                                      struct background * pba,
                                                      struct thermo* pth,
                                                      int* last_index_back,
                                                      double* pvecback){

  /** Summary: */

  /** Define local variables */
  double g_max;

  int index_tau;
  int index_tau_max;

  double tau;

  /** - find maximum of g */
  index_tau=pth->tt_size-1;
  while (pth->z_table[index_tau]>_Z_REC_MAX_) {
    index_tau--;
  }

  class_test(pth->thermodynamics_table[(index_tau+1)*pth->th_size+pth->index_th_g] >
             pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_g],
             pth->error_message,
             "The visibility function is not increasing at redshift _Z_REC_MAX_=%g, which is the value imposed in thermodynamics.h\n This implies that recombination must have already happened at a too early time.",_Z_REC_MAX_);

  while (pth->thermodynamics_table[(index_tau+1)*pth->th_size+pth->index_th_g] <
         pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_g]) {
    index_tau--;
  }

  g_max = pth->thermodynamics_table[index_tau*pth->th_size+pth->index_th_g];
  index_tau_max = index_tau;

  /* approximation for maximum of g, using cubic interpolation, assuming equally spaced z's */
  pth->z_rec=pth->z_table[index_tau+1]+0.5*(pth->z_table[index_tau+1]-pth->z_table[index_tau])
                                          *(pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_g]
                                                   -pth->thermodynamics_table[(index_tau+2)*pth->th_size+pth->index_th_g])
                                           /(pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_g]
                                                   -2.*pth->thermodynamics_table[(index_tau+1)*pth->th_size+pth->index_th_g]
                                                   +pth->thermodynamics_table[(index_tau+2)*pth->th_size+pth->index_th_g]
                                            );

  class_test(pth->z_rec+ppr->smallest_allowed_variation >= _Z_REC_MAX_,
             pth->error_message,
             "recombination (at z=%g) happens before _Z_REC_MAX_=%g, which is the maximum value imposed in thermodynamics.h",pth->z_rec+ppr->smallest_allowed_variation,_Z_REC_MAX_);

  class_test(pth->z_rec-ppr->smallest_allowed_variation <= _Z_REC_MIN_,
             pth->error_message,
             "recombination (at z=%g) happens after _Z_REC_MIN_=%g, which is the minimum value imposed in thermodynamics.h",pth->z_rec-ppr->smallest_allowed_variation,_Z_REC_MIN_);

  /** - find conformal recombination time using background_tau_of_z **/

  class_call(background_tau_of_z(pba,pth->z_rec,&(pth->tau_rec)),
             pba->error_message,
             pth->error_message);

  class_call(background_at_tau(pba,pth->tau_rec, pba->long_info, pba->inter_normal, last_index_back, pvecback),
             pba->error_message,
             pth->error_message);

  pth->rs_rec=pvecback[pba->index_bg_rs];
  pth->ds_rec=pth->rs_rec*pba->a_today/(1.+pth->z_rec);
  pth->da_rec=pvecback[pba->index_bg_ang_distance];
  pth->ra_rec=pth->da_rec*(1.+pth->z_rec)/pba->a_today;
  pth->angular_rescaling=pth->ra_rec/(pba->conformal_age-pth->tau_rec);

  /** - find damping scale at recombination (using linear interpolation) */
  if (pth->compute_damping_scale == _TRUE_) {

    pth->rd_rec = (pth->z_table[index_tau+1]-pth->z_rec)/(pth->z_table[index_tau+1]-pth->z_table[index_tau])*pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_r_d]
      +(pth->z_rec-pth->z_table[index_tau])/(pth->z_table[index_tau+1]-pth->z_table[index_tau])*pth->thermodynamics_table[(index_tau+1)*pth->th_size+pth->index_th_r_d];

  }

  /** - find time (always after recombination) at which tau_c/tau falls below some threshold, defining tau_free_streaming */
  class_call(background_tau_of_z(pba,pth->z_table[index_tau],&tau),
             pba->error_message,
             pth->error_message);

  while (1./pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_dkappa]/tau
         < ppr->radiation_streaming_trigger_tau_c_over_tau) {

    index_tau--;

    class_call(background_tau_of_z(pba,pth->z_table[index_tau],&tau),
               pba->error_message,
               pth->error_message);

  }

  pth->tau_free_streaming = tau;

  /** - find time above which visibility falls below a given fraction of its maximum */
  index_tau=index_tau_max;
  while ((pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_g] >
          g_max * ppr->neglect_CMB_sources_below_visibility)
         && (index_tau > 0))
    index_tau--;

  class_call(background_tau_of_z(pba,pth->z_table[index_tau],&(pth->tau_cut)),
             pba->error_message,
             pth->error_message);

  return _SUCCESS_;

}


/**
 * Calculate those quantities at the time of ending of baryon drag (It is precisely where tau_d crosses one)
 *
 * @param ppr                Input: pointer to precision structure
 * @param pba                Input: pointer to background structure
 * @param pth                Input/Output: pointer to initialized thermo structure
 * @param last_index_back    Input: temporary variable for storing index
 * @param pvecback           Input: pointer to some allocated pvecback
 * @return the error status
 */
int thermodynamics_calculate_drag_quantities(struct precision* ppr,
                                             struct background * pba,
                                             struct thermo* pth,
                                             int* last_index_back,
                                             double* pvecback){

  /** Summary: */

  /** Define local variables */
  int index_tau;

  /** - find baryon drag time (when tau_d crosses one, using linear interpolation) and sound horizon at that time */
  index_tau=0;
  while ((pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_tau_d] < 1.) && (index_tau < pth->tt_size)){
    index_tau++;
  }

  pth->z_d = pth->z_table[index_tau-1]+
    (1.-pth->thermodynamics_table[(index_tau-1)*pth->th_size+pth->index_th_tau_d])
    /(pth->thermodynamics_table[(index_tau)*pth->th_size+pth->index_th_tau_d]-pth->thermodynamics_table[(index_tau-1)*pth->th_size+pth->index_th_tau_d])
    *(pth->z_table[index_tau]-pth->z_table[index_tau-1]);

  class_call(background_tau_of_z(pba,pth->z_d,&(pth->tau_d)),
             pba->error_message,
             pth->error_message);

  class_call(background_at_tau(pba,pth->tau_d, pba->long_info, pba->inter_normal, last_index_back, pvecback),
             pba->error_message,
             pth->error_message);

  pth->rs_d=pvecback[pba->index_bg_rs];
  pth->ds_d=pth->rs_d*pba->a_today/(1.+pth->z_d);

  return _SUCCESS_;

}


/**
 * Print the output of the thermodynamics module
 *
 * @param pba   Input: pointer to background structure
 * @param pth   Input/Output: pointer to initialized thermo structure
 * @return the error status
 */
int thermodynamics_print_output(struct background* pba,
                                struct thermo* pth){

  /** Summary: */

  /** Define local variables */
  double tau_reio;

  /** - if verbose flag set to next-to-minimum value, print the main results */
  if (pth->thermodynamics_verbose > 0) {
    printf(" -> recombination at z = %f\n",pth->z_rec);
    printf("    corresponding to conformal time = %f Mpc\n",pth->tau_rec);
    printf("    with comoving sound horizon = %f Mpc\n",pth->rs_rec);
    printf("    angular diameter distance = %f Mpc\n",pth->da_rec);
    printf("    and sound horizon angle 100*theta_s = %f\n",100.*pth->rs_rec/pth->ra_rec);
    if (pth->compute_damping_scale == _TRUE_) {
      printf("    and with comoving photon damping scale = %f Mpc\n",pth->rd_rec);
      printf("    or comoving damping wavenumber k_d = %f 1/Mpc\n",2.*_PI_/pth->rd_rec);
    }
    printf(" -> baryon drag stops at z = %f\n",pth->z_d);
    printf("    corresponding to conformal time = %f Mpc\n",pth->tau_d);
    printf("    with comoving sound horizon rs = %f Mpc\n",pth->rs_d);
    if ((pth->reio_parametrization == reio_camb) || (pth->reio_parametrization == reio_half_tanh)) {
      if (pth->reio_z_or_tau==reio_tau){
        printf(" -> reionization at z = %f\n",pth->z_reio);
      }
      if (pth->reio_z_or_tau==reio_z){
        printf(" -> reionization with optical depth = %f\n",pth->tau_reio);
      }
      class_call(background_tau_of_z(pba,pth->z_reio,&tau_reio),
                 pba->error_message,
                 pth->error_message);
      printf("    corresponding to conformal time = %f Mpc\n",tau_reio);
    }
    if (pth->reio_parametrization == reio_bins_tanh) {
      printf(" -> binned reionization gives optical depth = %f\n",pth->tau_reio);
    }
    if (pth->reio_parametrization == reio_many_tanh) {
      printf(" -> many-step reionization gives optical depth = %f\n",pth->tau_reio);
    }
    if (pth->reio_parametrization == reio_inter) {
      printf(" -> interpolated reionization history gives optical depth = %f\n",pth->tau_reio);
    }
    if (pth->thermodynamics_verbose > 1) {
      printf(" -> free-streaming approximation can be turned on as soon as tau=%g Mpc\n",pth->tau_free_streaming);
    }
  }

  return _SUCCESS_;

}



/**
 * This subroutine contains the reionization function \f$ X_e(z) \f$ (one for each scheme) and gives x and dx for a given z.
 *
 * @param z     Input: redshift
 * @param pth   Input: pointer to thermo structure, to know which scheme is used
 * @param preio Input: pointer to reionization parameters of the function \f$ X_e(z) \f$
 * @param x     Output: \f$ X_e(z) \f$
 * @param dx    Output: \f$ dX_e(z)/dz \f$
 */
int thermodynamics_reionization_function(double z,
                                         struct thermo * pth,
                                         struct thermo_reionization_parameters * preio,
                                         double * x,
                                         double * dx){

  /** Summary: */

  /** - define local variables */
  double argument,dargument;
  int i;
  double z_jump;

  int jump;
  double center,before, after,width,one_jump;
  double z_min, z_max;


  /** - implementation of ionization function similar to the one in CAMB */
  if ((pth->reio_parametrization == reio_camb) || (pth->reio_parametrization == reio_half_tanh)) {

    /** - --> case z > z_reio_start */
    if (z > preio->reionization_parameters[preio->index_reio_start]) {
      *x = preio->reionization_parameters[preio->index_reio_xe_before];
      *dx = 0.0;
    }
    else {
      /** - --> case z < z_reio_start: hydrogen contribution (tanh of complicated argument) */
      argument = (pow((1.+preio->reionization_parameters[preio->index_reio_redshift]),
                    preio->reionization_parameters[preio->index_reio_exponent])
                   -pow((1.+z),preio->reionization_parameters[preio->index_reio_exponent]))
                 /(preio->reionization_parameters[preio->index_reio_exponent]
          /* no possible segmentation fault: checked to be non-zero in thermodynamics_reionization */
                 *pow((1.+preio->reionization_parameters[preio->index_reio_redshift]),
                    (preio->reionization_parameters[preio->index_reio_exponent]-1.)))
        /preio->reionization_parameters[preio->index_reio_width];
      /* no possible segmentation fault: checked to be non-zero in thermodynamics_reionization */

      dargument = -pow((1.+z),(preio->reionization_parameters[preio->index_reio_exponent]-1.))
          /* no possible segmentation fault: checked to be non-zero in thermodynamics_reionization */
                 /pow((1.+preio->reionization_parameters[preio->index_reio_redshift]),
                   (preio->reionization_parameters[preio->index_reio_exponent]-1.))
                 /preio->reionization_parameters[preio->index_reio_width];

      if (pth->reio_parametrization == reio_camb) {
        *x = (preio->reionization_parameters[preio->index_reio_xe_after]
               -preio->reionization_parameters[preio->index_reio_xe_before])
             *(tanh(argument)+1.)/2.
             +preio->reionization_parameters[preio->index_reio_xe_before];

        *dx = (preio->reionization_parameters[preio->index_reio_xe_after]
               -preio->reionization_parameters[preio->index_reio_xe_before])
              *(1-tanh(argument)*tanh(argument))/2.*dargument;
      }
      else {
        *x = (preio->reionization_parameters[preio->index_reio_xe_after]
               -preio->reionization_parameters[preio->index_reio_xe_before])
             *tanh(argument)
             +preio->reionization_parameters[preio->index_reio_xe_before];

        *dx = (preio->reionization_parameters[preio->index_reio_xe_after]
                -preio->reionization_parameters[preio->index_reio_xe_before])
              *(1-tanh(argument)*tanh(argument))*dargument;
      }

      /** - --> case z < z_reio_start: helium contribution (tanh of simpler argument) */
      if (pth->reio_parametrization == reio_camb) {
        argument = (preio->reionization_parameters[preio->index_helium_fullreio_redshift] - z)
          /preio->reionization_parameters[preio->index_helium_fullreio_width];

        dargument = -1./preio->reionization_parameters[preio->index_helium_fullreio_width];
        /* no possible segmentation fault: checked to be non-zero in thermodynamics_reionization */
        *x += preio->reionization_parameters[preio->index_helium_fullreio_fraction]
              *(tanh(argument)+1.)/2.;

        *dx += preio->reionization_parameters[preio->index_helium_fullreio_fraction]
               *(1-tanh(argument)*tanh(argument))/2.*dargument;
      }
    }

    return _SUCCESS_;

  }

  /** - implementation of binned ionization function similar to astro-ph/0606552 */
  if (pth->reio_parametrization == reio_bins_tanh) {

    /** - --> case z > z_reio_start */
    if (z > preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1]) {
      *x = preio->reionization_parameters[preio->index_reio_xe_before];
      *dx = 0.0;

    }
    else if (z < preio->reionization_parameters[preio->index_reio_first_z]) {
      *x = preio->reionization_parameters[preio->index_reio_first_xe];
      *dx = 0.0;
    }
    else {
      i = 0;
      while (preio->reionization_parameters[preio->index_reio_first_z+i+1]<z) i++;

      /* fix the final xe to xe_before*/
      preio->reionization_parameters[preio->index_reio_first_xe+preio->reio_num_z-1] = preio->reionization_parameters[preio->index_reio_xe_before];

      /* This is the expression of the tanh-like jumps of the reio_bins_tanh scheme until the 10.06.2015. It appeared to be
         not robust enough. It could lead to a kink in xe(z) near the maximum value of z at which reionisation is sampled. It has
         been replaced by the simpler and more robust expression below.

      *xe = preio->reionization_parameters[preio->index_reio_first_xe+i]
        +0.5*(tanh((2.*(z-preio->reionization_parameters[preio->index_reio_first_z+i])
                    /(preio->reionization_parameters[preio->index_reio_first_z+i+1]
                      -preio->reionization_parameters[preio->index_reio_first_z+i])-1.)
                   /preio->reionization_parameters[preio->index_reio_step_sharpness])
              /tanh(1./preio->reionization_parameters[preio->index_reio_step_sharpness])+1.)
        *(preio->reionization_parameters[preio->index_reio_first_xe+i+1]
          -preio->reionization_parameters[preio->index_reio_first_xe+i]);
      */

      /* compute the central redshift value of the tanh jump */
      if(i == preio->reio_num_z-2){
        z_jump = preio->reionization_parameters[preio->index_reio_first_z+i]
          + 0.5*(preio->reionization_parameters[preio->index_reio_first_z+i]
                 -preio->reionization_parameters[preio->index_reio_first_z+i-1]);
      }
      else{
        z_jump =  0.5*(preio->reionization_parameters[preio->index_reio_first_z+i+1]
                       + preio->reionization_parameters[preio->index_reio_first_z+i]);
      }

      /* implementation of the tanh jump */
      *x = preio->reionization_parameters[preio->index_reio_first_xe+i]
        +0.5*(tanh((z-z_jump)
                   /preio->reionization_parameters[preio->index_reio_step_sharpness])+1.)
        *(preio->reionization_parameters[preio->index_reio_first_xe+i+1]
          -preio->reionization_parameters[preio->index_reio_first_xe+i]);

      *dx = 0.5*(1-tanh((z-z_jump)/preio->reionization_parameters[preio->index_reio_step_sharpness])
                  *tanh((z-z_jump)/preio->reionization_parameters[preio->index_reio_step_sharpness]))
            *(preio->reionization_parameters[preio->index_reio_first_xe+i+1]
              -preio->reionization_parameters[preio->index_reio_first_xe+i])
            /preio->reionization_parameters[preio->index_reio_step_sharpness];
    }

    return _SUCCESS_;

  }

  /** - implementation of many tanh jumps */
  if(pth->reio_parametrization == reio_many_tanh){

    /** - --> case z > z_reio_start */
    if(z > preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1]){
      *x = preio->reionization_parameters[preio->index_reio_xe_before];
      *dx = 0.0;
    }
    else if(z > preio->reionization_parameters[preio->index_reio_first_z]){

      *x = preio->reionization_parameters[preio->index_reio_xe_before];
      *dx = 0.0;

      /* fix the final xe to xe_before*/
      preio->reionization_parameters[preio->index_reio_first_xe+preio->reio_num_z-1] = preio->reionization_parameters[preio->index_reio_xe_before];

      for (jump=1; jump<preio->reio_num_z-1; jump++){

        center = preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1-jump];

        /* before and after are meant with respect to growing z, not growing time */
        before = preio->reionization_parameters[preio->index_reio_first_xe+preio->reio_num_z-1-jump]
          -preio->reionization_parameters[preio->index_reio_first_xe+preio->reio_num_z-jump];
        after = 0.;
        width = preio->reionization_parameters[preio->index_reio_step_sharpness];

        one_jump = before + (after-before)*(tanh((z-center)/width)+1.)/2.;

        *x += one_jump;
        *dx += (after-before)*(1-tanh((z-center)/width)*tanh((z-center)/width))/2./width;
      }

    }
    else{
      *x = preio->reionization_parameters[preio->index_reio_first_xe];
      *dx = 0.0;
    }

    return _SUCCESS_;

  }

  /** - implementation of reio_inter */
  if (pth->reio_parametrization == reio_inter) {

    /** - --> case z > z_reio_start */
    if (z > preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1]){
      *x = preio->reionization_parameters[preio->index_reio_xe_before];
      *dx = 0.0;
    }
    else{
      i=0;
      while (preio->reionization_parameters[preio->index_reio_first_z+i+1] < z) i++;

      z_min = preio->reionization_parameters[preio->index_reio_first_z+i];
      z_max = preio->reionization_parameters[preio->index_reio_first_z+i+1];

      /* fix the final xe to xe_before*/
      preio->reionization_parameters[preio->index_reio_first_xe+preio->reio_num_z-1] = preio->reionization_parameters[preio->index_reio_xe_before];


      class_test(z<z_min,
                 pth->error_message,
                 "z out of range for reionization interpolation");

      class_test(z>z_max,
                 pth->error_message,
                 "z out of range for reionization interpolation");

      argument =(z-preio->reionization_parameters[preio->index_reio_first_z+i])
        /(preio->reionization_parameters[preio->index_reio_first_z+i+1]
          -preio->reionization_parameters[preio->index_reio_first_z+i]);

      dargument = 1./(preio->reionization_parameters[preio->index_reio_first_z+i+1]
          -preio->reionization_parameters[preio->index_reio_first_z+i]);

      *x = preio->reionization_parameters[preio->index_reio_first_xe+i]
        + argument*(preio->reionization_parameters[preio->index_reio_first_xe+i+1]
             -preio->reionization_parameters[preio->index_reio_first_xe+i]);

      *dx = dargument*(preio->reionization_parameters[preio->index_reio_first_xe+i+1]
             -preio->reionization_parameters[preio->index_reio_first_xe+i]);

      class_test(*x<0.,
                 pth->error_message,
                 "Interpolation gives negative ionization fraction\n",
                 argument,
                 preio->reionization_parameters[preio->index_reio_first_xe+i],
                 preio->reionization_parameters[preio->index_reio_first_xe+i+1]);
    }

    return _SUCCESS_;

  }

  class_test(0 == 0,
             pth->error_message,
             "value of reio_parametrization=%d unclear",pth->reio_parametrization);

}


/**
 * Integrate thermodynamics with RECFAST or HYREC.
 *
 * Integrate thermodynamics with RECFAST, allocate and fill the part of the thermodynamics interpolation table (the rest is filled in
 * thermodynamics_init). Called once by thermodynamics_recombination, from thermodynamics_init.
 *
 * Version modified by Daniel Meinert and Nils Schoeneberg to use the ndf15 evolver or any other evolver inherent to CLASS,
 * modified again by Nils Schoeneberg to use wrappers
 *
 *********************************************************************************************************************************
 * RECFAST is an integrator for Cosmic Recombination of Hydrogen and Helium, developed by Douglas Scott (dscott@astro.ubc.ca)
 * based on calculations in the paper Seager, Sasselov & Scott (ApJ, 523, L1, 1999) and "fudge" updates in Wong, Moss & Scott (2008).
 *
 * Permission to use, copy, modify and distribute without fee or royalty at any tier, this software and its documentation, for any
 * purpose and without fee or royalty is hereby granted, provided that you agree to comply with the following copyright notice and
 * statements, including the disclaimer, and that the same appear on ALL copies of the software and documentation,
 * including modifications that you make for internal use or for distribution:
 *
 * Copyright 1999-2010 by University of British Columbia.  All rights reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", AND U.B.C. MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.
 * BY WAY OF EXAMPLE, BUT NOT LIMITATION, U.B.C. MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANTABILITY
 * OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE OR DOCUMENTATION WILL NOT INFRINGE
 * ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS.
 *********************************************************************************************************************************
 *
 * Version 1.5: includes extra fitting function from Rubino-Martin et al. arXiv:0910.4383v1 [astro-ph.CO]
 *
 * @param ppr      Input: pointer to precision structure
 * @param pba      Input: pointer to background structure
 * @param pth      Input: pointer to thermodynamics structure
 * @param ptw      Output: pointer to thermo_workspace structure
 * @param pvecback Input: pointer to an allocated (but empty) vector of background variables
 * @return the error status
 *
 * Integrate thermodynamics with your favorite recombination code. The default options are HyRec and Recfast.
 */
int thermodynamics_solve(struct precision * ppr,
                         struct background * pba,
                         struct thermo * pth,
                         struct thermo_workspace * ptw,
                         double * pvecback){
  /** Summary: */

  /** - define local variables */
  /* Index of current approximation */
  int index_ap;
  /* number of time intervals of one approximation scheme*/
  int interval_number;
  /* index running over such time intervals */
  int index_interval;
  /* edge of intervals where approximation scheme is uniform: z_ini, z_switch_1, ..., z_end */
  double * interval_limit;
  /* other z sampling variables */
  int i;
  double * mz_output;

  /* contains all fixed parameters which should be passed to thermodynamics_solve_derivs */
  struct thermodynamics_parameters_and_workspace tpaw;

  /* function pointer to ODE evolver and names of possible evolvers */
  extern int evolver_rk();
  extern int evolver_ndf15();
  int (*generic_evolver)();

  class_call(thermodynamics_set_parameters_reionization(ppr,
                                                        pba,
                                                        pth,
                                                        ptw->ptrp),
             pth->error_message,
             pth->error_message);

  /** - define the fields of the 'thermodynamics parameter and workspace' structure */
  tpaw.pba = pba;
  tpaw.ppr = ppr;
  tpaw.pth = pth;
  tpaw.pvecback = pvecback;
  tpaw.ptw = ptw;

  /** - define time sampling in minus z */
  /* -> Create mz_output array of minus z (from mz=-zinitial growing towards mz=0) */
  class_alloc(mz_output,pth->tt_size*sizeof(double), pth->error_message);
  for(i=0; i < pth->tt_size; ++i){
    mz_output[i] = -pth->z_table[pth->tt_size-1-i];
  }

  /** - define switching intervals */
  /* -> Create interval limits */
  class_alloc(interval_limit,(ptw->ptdw->ap_size+1)*sizeof(double),pth->error_message);
  /* fix interval number to number of approximations */
  interval_number = ptw->ptdw->ap_size;
  /* integration starts at z_ini and ends at z_end */
  interval_limit[0]= mz_output[0];
  interval_limit[ptw->ptdw->ap_size] = mz_output[ptw->Nz_tot-1];
  /* each interval ends with the proper ending redshift of its approximation */
  for(index_ap=0; index_ap < ptw->ptdw->ap_size-1; index_ap++){
    interval_limit[index_ap+1] = -ptw->ptdw->ap_z_limits[index_ap];
  }

  /** - loop over intervals over which approximation scheme is uniform. For each interval: */
  for (index_interval=0; index_interval<interval_number; index_interval++) {

    /** - --> (a) fix current approximation scheme. */
    ptw->ptdw->ap_current = index_interval;

    /** - --> (b) define the vector of quantities to be integrated over. If the current interval starts from the initial time
                  zinitial, fill the vector with initial conditions for. If it starts from an approximation switching point,
                  redistribute correctly the values from the previous to the new vector. */
    class_call(thermodynamics_vector_init(ppr,
                                          pba,
                                          pth,
                                          interval_limit[index_interval],
                                          ptw),
               pth->error_message,
               pth->error_message);

    /** - --> (c) integrate the quantities over the current interval. */
    if(ppr->evolver == rk){
      generic_evolver = evolver_rk;
    }
    else{
      generic_evolver = evolver_ndf15;
    }

    /* If we have the optical depth tau_reio as input the last evolver step (reionization approximation) is done separately in a loop,
       to approximate the redshift of reionization through the input of tau_reio. This is done using a bisection method
       It is similar to CLASS's shooting, just that we only re-do this last approximation step of the thermodynamics calculation */
    if(pth->reio_z_or_tau == reio_tau && index_interval == ptw->ptdw->index_ap_reio){
        class_call(thermodynamics_reionization_evolve_with_tau(&tpaw,
                                                               interval_limit[index_interval],
                                                               interval_limit[index_interval+1],
                                                               mz_output,
                                                               pth->tt_size),
                   pth->error_message,
                   pth->error_message);
    }
    else{
      class_call(generic_evolver(thermodynamics_solve_derivs,
                                 interval_limit[index_interval],
                                 interval_limit[index_interval+1],
                                 ptw->ptdw->tv->y,
                                 ptw->ptdw->tv->used_in_output,
                                 ptw->ptdw->tv->tv_size,
                                 &tpaw,
                                 ppr->tol_thermo_integration,
                                 ppr->smallest_allowed_variation,
                                 thermodynamics_solve_timescale,  // timescale
                                 1., // stepsize
                                 mz_output, // values of z for output
                                 pth->tt_size, // size of previous array
                                 thermodynamics_solve_store_sources, // function for output
                                 NULL, // print variables
                                 pth->error_message),
                   pth->error_message,
                   pth->error_message);
    }

  }

  /** - Compute reionization optical depth, if not supplied as input parameter */
  if (pth->reio_parametrization != reio_none && pth->reio_z_or_tau == reio_z) {

    class_call(thermodynamics_reionization_get_tau(ppr,
                                                   pba,
                                                   pth,
                                                   ptw),
               pth->error_message,
               pth->error_message);

    pth->tau_reio=ptw->reionization_optical_depth;

  }

  /** - free quantities allocated at the beginning of the routine */
  if(ptw->ptdw->ap_size != 0){
    class_call(thermodynamics_vector_free(ptw->ptdw->tv),
               pth->error_message,
               pth->error_message);
  }

  free(interval_limit);
  free(mz_output);

  return _SUCCESS_;

}


/**
 * Subroutine evaluating the derivative with respect to negative redshift of thermodynamical quantities (from RECFAST version 1.4,
 * modified by Daniel Meinert and Nils Schoeneberg)
 *
 * Automatically recognizes the current approximation interval and computes the needed derivatives for this interval:
 * \f$ d T_{mat} / dz , d x_H / dz, d x_{He} / dz \f$.
 *
 * This is one of the few functions in the code which are passed to the generic_evolver routine.  Since generic_evolver
 * should work with functions passed from various modules, the format of the arguments is a bit special:
 *
 * - fixed parameters and workspaces are passed through a generic pointer. Here, this pointer contains the precision, background
 *   and thermo structures, plus a background vector, but generic_evolver doesn't know its precise structure.
 *
 * - the error management is a bit special: errors are not written as usual to pth->error_message, but to a generic error_message
 *   passed in the list of arguments.
 *
 * @param z                        Input: redshift
 * @param y                        Input: vector of variable to integrate
 * @param dy                       Output: its derivative (already allocated)
 * @param parameters_and_workspace Input: pointer to fixed parameters (e.g. indices) and workspace (already allocated)
 * @param error_message            Output: error message
 */
int thermodynamics_solve_derivs(double mz,
                                double * y,
                                double * dy,
                                void * parameters_and_workspace,
                                ErrorMsg error_message) {

  /** Summary: */

  /** Define local variables */
  double z;
  int index_y;
  /* Shorthand notations */
  double x,nH,Trad,Tmat,x_H,x_He,dx,dxdlna,Hz,eps,depsdlna,dHdlna;
  /* Photon-Baryon interaction rate */
  double R_g;
  /*Interacting dark matter - baryons expressions DCH*/
  double T_idm_b;

  /* Heat capacity of the IGM */
  double heat_capacity;
  /* Background structure */
  double tau;
  int last_index_back;

  /* Shorthand notations for all of the structs */
  struct thermodynamics_parameters_and_workspace * ptpaw;
  struct precision * ppr;
  struct background * pba;
  struct thermo * pth;
  double * pvecback;
  struct thermo_workspace * ptw;
  struct thermo_diffeq_workspace * ptdw;
  struct thermo_vector * ptv;
  struct thermorecfast * precfast;
  struct thermohyrec * phyrec;
  struct heating* phe;
  int ap_current;

  /* Redshift */
  z = -mz;

  /** - rename structure fields (just to avoid heavy notations) */
  /* Structs */
  ptpaw = parameters_and_workspace;
  ppr = ptpaw->ppr;
  pba = ptpaw->pba;
  pth = ptpaw->pth;
  phe = &(pth->he);
  /* pvecback */
  pvecback = ptpaw->pvecback;
  /* Thermo workspace & vector */
  ptw = ptpaw->ptw;
  ptdw = ptw->ptdw;
  ptv = ptdw->tv;
  /* Recfast/HyRec */
  precfast = ptdw->precfast;
  phyrec = ptdw->phyrec;
  /* Approximation flag */
  ap_current = ptdw->ap_current;

  /** - get background/thermo quantities in this point */
  class_call(background_tau_of_z(pba,
                                 z,
                                 &tau),
             pba->error_message,
             error_message);

  class_call(background_at_tau(pba,
                               tau,
                               pba->long_info,
                               pba->inter_normal,
                               &last_index_back,
                               pvecback),
             pba->error_message,
             error_message);

  /** Hz is H in inverse seconds (while pvecback returns [H0/c] in inverse Mpcs). Modify these for some non-trivial
      background evolutions or CMB temperature changes */
  Hz = pvecback[pba->index_bg_H] * _c_ / _Mpc_over_m_;
  nH = ptw->SIunit_nH0 * (1.+z) * (1.+z) * (1.+z);
  Trad = ptw->Tcmb * (1.+z);

  /** - Matter temperature and ionization fraction */
  class_call(thermodynamics_solve_current_quantities(z,y,pth,ptw,ap_current),
             pth->error_message,
             pth->error_message);

  /** - Calculate quantities for interacting dark matter with baryons DCH */
  if(pth->has_idm_b == _TRUE_){
    class_call(thermodynamics_solve_current_idm_b(pba,z,y,dy,pth,ptw,pvecback),
               pth->error_message,
               pth->error_message);
  }

  /* Save the output in local variables */
  x = ptdw->x_reio;
  x_H = ptdw->x_H;
  x_He = ptdw->x_He;
  Tmat = ptdw->Tmat;
  if(pth->has_idm_b == _TRUE_){
    T_idm_b = ptdw->T_idm_b; //DCH
  }

  /** - Calculate heating */
  /* in case of energy injection, we currently neglect the contribution to helium ionization ! */
  /* Calculate the energy injection INCLUDING reionization ! */
  class_call(heating_calculate_at_z(pba,pth,x,z,Tmat,pvecback),
             phe->error_message,
             error_message);

  /* The following derivatives for x_H and x_He should be calculated using only the x from recombination.
     Of course, the full treatment would involve the actual evolution equations for x_H and x_He during reionization,
     but these are not yet fully implemented. */
  x = ptdw->x_noreio;

  /** - HyRec */
  if(pth->recombination == hyrec && phyrec->to_store == _TRUE_){
    class_call(thermodynamics_hyrec_calculate_xe(pth,phyrec,z,Hz,Tmat,Trad,&x,&dxdlna),
               phyrec->error_message,
               error_message);
    x = ptdw->x_reio;
  }

  /** - RecfastCLASS */
  if(pth->recombination == recfast){
    /** - Hydrogen equations */
    if(ptdw->require_H){
      class_call(thermodynamics_recfast_dx_H_dz(pth,precfast,x_H,x,nH,z,Hz,Tmat,Trad,&(dy[ptv->index_x_H])),
                 precfast->error_message,
                 error_message);
    }

    /** - Helium equations */
    if(ptdw->require_He){
      class_call(thermodynamics_recfast_dx_He_dz(pth,precfast,x_He,x,x_H,nH,z,Hz,Tmat,Trad,&(dy[ptv->index_x_He])),
                 precfast->error_message,
                 error_message);
    }
  }

  /* Get dxdlna */
  class_call(thermodynamics_solve_current_dxdlna(z,y,dy,pth,ptw,ap_current),
             pth->error_message,
             pth->error_message);
  dxdlna = ptdw->dxdlna;

  /** - Matter temperature equations */
  /* Tmat is always integrated */
  R_g = ( 2. * _sigma_/_m_e_/_c_ ) * ( 4./3. * pvecback[pba->index_bg_rho_g] * _Jm3_over_Mpc2_ );
  heat_capacity = (3./2.)*_k_B_*nH*(1.+ptw->fHe+x);

 /**
  * A note on the temperature definition:
  *
  * All equations work with D_Tmat = Tmat - Trad
  *
  * Thus all derivatives are calculated as dD_Tmat/dz = dTmat/dz - Tcmb
  **/

 /**
  * A note on the 'early' time steady state expansion (activated here before HeIII recombination):
  *
  * Note: dTr/dz = Tr/(1+z) = Tcmb
  *
  * The early system of CMB and matter is very tightly coupled anyway, so we can expand in the following way:
  * The full equation is dTm/dz = (Tm-Tr)/e /(1+z) + 2 Tm/(1+z). Here e = H*(1+x+f)/(cT*Tr^4*x) << 1 at early times
  *
  * Find the first order solution in e, by multiplying in (1+z)*e, and approximate
  *  e*(dTm/dz)*(1+z) ~ e*(dTr/dz)*(1+z) + O(e^2) ~ e * Tr
  *
  * You find e*Tr = (Tm-Tr) + 2 Tm * e
  * Thus Tm = (1+2*e)/(1+e) * Tr = Tr/(1+e) + O(e^2)
  *
  * This is the steady state solution, which is the SAME as e.g. in HyRec
  * In our notation, eps = e*Tr, so we get Tm = Tr - eps
  *
  * So, taking the derivative of the right hand side, we obtain dTm/dz = Tcmb - eps*(dln(eps)/dz)
  *
  * Now use the form of eps = Tr*e = H*(1+x+f)/(cT*Tr^3*x) to derive the remaining terms in the below formula
  * => dln(eps)/dlna = dln(H)/dlna  - (1+f)/(1+x+f)*dln(x)/dlna + 3*dln(Tr)/dlna
  * => dln(eps)/dz = -1/(1.+z)*dln(eps)/dlna
  *
  * With Tr ~ a^(-1) and dD_Tmat/dz = dTmat/dz - Tcmb , you find the terms as below
  **/

  if( ap_current == ptdw->index_ap_brec){
    dHdlna = (1.+z)*pvecback[pba->index_bg_H_prime]/pvecback[pba->index_bg_H]/pba->a_today * _c_ / _Mpc_over_m_;
    eps =  Trad * Hz * (1.+x+ptw->fHe) / (R_g*x);
    depsdlna = (1.+ptw->fHe)/(1.+ptw->fHe+x)*(dxdlna/x) - dHdlna/Hz - 3.;
    /* v 1.5: like in camb, add here a smoothing term as suggested by Adam Moss */
    dy[ptdw->tv->index_D_Tmat] = -1./(1.+z)*eps*depsdlna;
  }

  else {
    /* Full equations at later times */
    dy[ptv->index_D_Tmat] =
        + 2.*Tmat/(1.+z)                                                          /* Adiabatic expansion */
        + R_g * x / (1.+x+ptw->fHe) * (Tmat-Trad) / (Hz*(1.+z))                   /* Coupling to photons*/
        - phe->pvecdeposition[phe->index_dep_heat] / heat_capacity / (Hz*(1.+z))  /* Heating from energy injection */
        - ptw->Tcmb;                                                              /* dTrad/dz */

    /* Add term coming from interacting Dark Matter - baryons */
    if(pth->has_idm_b == _TRUE_){
      dy[ptv->index_D_Tmat] += 2.*_m_p_/(pth->m_idm*_eV_/(_c_*_c_)+_m_p_)*ptdw->R_idm_b*(Tmat-T_idm_b) / (pvecback[pba->index_bg_a]*pvecback[pba->index_bg_H]*(1.+z));
    }
  }

  /** - Dark Matter temperature equations */
  /* T_idm_b is always integrated */

  if(pth->has_idm_b == _TRUE_){
    dy[ptv->index_T_idm_b] =
      + 2.*T_idm_b/(1.+z)
      + 2.*(pth->m_idm*_eV_/(_c_*_c_))/((pth->m_idm*_eV_/(_c_*_c_))+_m_p_)*ptdw->R_idm_b*(T_idm_b-Tmat) / (pvecback[pba->index_bg_a]*pvecback[pba->index_bg_H]*(1.+z));
  }


  /*
   * If we have extreme heatings, recombination does not fully happen
   * and/or re-ionization happens before a redshift of reionization_z_start_max (default = 50).
   *
   * */
  class_test(x>1.0 && ap_current != ptdw->index_ap_reio && z < ppr->z_end_reco_test,
             pth->error_message,
             "At redshift %.5g : Recombination did not complete by redshift %.5g, or re-ionization happened before %.5g.\nIf this is a desired behavior, please adjust z_end_reco_test and/or reionization_z_start_max.",
             z,ppr->z_end_reco_test,ppr->reionization_z_start_max);


  /* time-invert derivatives (As the evolver evolves with -z, not with +z) */
  for(index_y=0;index_y<ptdw->tv->tv_size;index_y++){
    dy[index_y]=-dy[index_y];
  }

  return _SUCCESS_;

}

/**
 * This routine computes for the different codes and approximations
 * the quantities x and Tmat and sets them within the workspace
 *
 * @param z            Input: redshift
 * @param y            Input: vector of evolver quantities
 * @param pth          Input: pointer to thermodynamics structure
 * @param ptw          Input/Output: pointer to thermo workspace
 * @param current_ap   Input: index of the wished approximation scheme
 * @return the error status
 */
int thermodynamics_solve_current_quantities(double z,
                                            double * y,
                                            struct thermo * pth,
                                            struct thermo_workspace * ptw,
                                            int current_ap){

  /** Summary: */

  /** Define local variables */
  struct thermo_diffeq_workspace * ptdw = ptw->ptdw;
  struct thermorecfast * precfast = ptw->ptdw->precfast;
  struct thermohyrec * phyrec = ptw->ptdw->phyrec;
  struct thermo_vector * ptv = ptdw->tv;

  /* Thermo quantities */
  double x_H, x_He, x, Tmat;
  /* Analytical quantities */
  double rhs, sqrt_val, drhs_dlna;
  /* Temporary quantities */
  double dx;

  /** Set Tmat from the evolver (it is always evolved). */
  Tmat = y[ptv->index_D_Tmat] + ptw->Tcmb*(1.+z);

  /* Note, dxdlna is always calculated from a steady-state solution of Tmat in 0th order,
   * i.e. approximating dlnTmat_dlna = dlnTrad_dlna = -1. */
  double dlnTmat_dlna = -(1.+z)*(ptv->dy[ptv->index_D_Tmat] + ptw->Tcmb)/Tmat;//-1.; /* Steady state approximation to 0th order */

  /** Case RecfastCLASS :: */
  if(pth->recombination == recfast){

    /** Set the ionization fractions x_H, x_He and x for each regime. */
    /** - --> first regime: H and Helium fully ionized */
    if(current_ap == ptdw->index_ap_brec){

      /* analytic approximations */
      rhs = pth->n_e/exp( 1.5*log(precfast->CR*ptdw->Tmat/(1.+z)/(1.+z)) - precfast->CB1_He2/ptdw->Tmat );
      sqrt_val = sqrt(pow(1.-rhs*(1.+ptw->fHe),2) + 4.*rhs*(1.+2*ptw->fHe));

      x = 2.*(1+2.*ptw->fHe)/(1.-rhs*(1.+ptw->fHe) + sqrt_val);

      ptdw->x_H = 1.;
      ptdw->x_He = 1.;

      ptdw->dxdlna = 0.;

    }
    /** - --> second regime: first Helium recombination (analytic approximation) */
    else if (current_ap == ptdw->index_ap_He1) {

      /* analytic approximations */
      rhs = exp( 1.5*log(precfast->CR*Tmat/(1.+z)/(1.+z)) - precfast->CB1_He2/Tmat ) / ptw->SIunit_nH0;
      sqrt_val = sqrt(pow((rhs-1.-ptw->fHe),2) + 4.*(1.+2.*ptw->fHe)*rhs);

      x = 0.5*(sqrt_val - (rhs-1.-ptw->fHe));

      ptdw->x_H = 1.;
      ptdw->x_He = 1.;

      drhs_dlna = rhs*((precfast->CB1_He2*dlnTmat_dlna/ptdw->Tmat)+1.5*(dlnTmat_dlna+2.) );
      ptdw->dxdlna = 0.5*(  ((rhs-1.-ptw->fHe) + 2.*(1.+2.*ptw->fHe))/sqrt_val   -   1.  )*drhs_dlna;

    }
    /** - --> third regime: first Helium recombination finished */
    else if (current_ap == ptdw->index_ap_He1f) {

      /* analytic approximations */
      rhs = 0.25*pth->n_e/exp( 1.5*log(precfast->CR*ptdw->Tmat/(1.+z)/(1.+z)) - precfast->CB1_He1/ptdw->Tmat );
      sqrt_val = sqrt(pow(1.-rhs,2) + 4.*rhs*(1.+ptw->fHe));

      x = 2.*(1+ptw->fHe)/(1.-rhs + sqrt_val);

      ptdw->x_H = 1.;
      ptdw->x_He = 1.;

      ptdw->dxdlna = 0.;

    }
    /** - --> fourth regime: second Helium recombination starts */
    else if (current_ap == ptdw->index_ap_He2) {

      /* analytic approximations */
      rhs = 4.*exp(1.5*log(precfast->CR*Tmat/(1.+z)/(1.+z)) - precfast->CB1_He1/Tmat ) / ptw->SIunit_nH0;
      sqrt_val = sqrt(pow((rhs-1.),2) + 4.*(1.+ptw->fHe)*rhs );

      x = 0.5*(sqrt_val - (rhs-1.));

      ptdw->x_H = 1.;
      ptdw->x_He = (x-1.)/ptw->fHe;

      drhs_dlna = rhs*((precfast->CB1_He1*dlnTmat_dlna/ptdw->Tmat)+1.5*(dlnTmat_dlna+2.) );
      ptdw->dxdlna = 0.5*(  ((rhs-1.) + 2.*(1.+ ptw->fHe))/sqrt_val   -   1.  )*drhs_dlna;

    }
    /** - --> fifth regime: Hydrogen recombination starts */
    else if (current_ap == ptdw->index_ap_H) {

      /* analytic approximations */
      rhs = exp(1.5*log(precfast->CR*Tmat/(1.+z)/(1.+z)) - precfast->CB1/Tmat)/ptw->SIunit_nH0;
      sqrt_val = sqrt(pow(rhs,2)+4.*rhs);

      x_H = 0.5*(sqrt_val - rhs);
      x_He = y[ptv->index_x_He];
      x = x_H + ptw->fHe * x_He;

      ptdw->x_H = x_H;
      ptdw->x_He = x_He;

      /* dxdlna will be set later */

    }
    /** - --> sixth regime: full recombination */
    else if (current_ap == ptdw->index_ap_frec) {

      x_H = y[ptv->index_x_H];
      x_He = y[ptv->index_x_He];
      x = x_H + ptw->fHe * x_He;

      ptdw->x_H = x_H;
      ptdw->x_He = x_He;

      /* dxdlna will be set later */

    }
    /** - --> sixth regime: reionization */
    else if (current_ap == ptdw->index_ap_reio) {

      x_H = y[ptv->index_x_H];
      x_He = y[ptv->index_x_He];
      x = x_H + ptw->fHe * x_He;

      ptdw->x_H = x_H;
      ptdw->x_He = x_He;

      /* dxdlna will be set later */

    }
  }
  /** Case HyRec :: */
  else{
    if(current_ap == ptdw->index_ap_brec){
      x = 1. + 2.*ptw->fHe;
      ptdw->dxdlna = 0.;
    }
    else{
      class_call(thermodynamics_hyrec_get_xe(phyrec,z,&x,&(ptdw->dxdlna)),
                 phyrec->error_message,
                 pth->error_message);
    }
  }

  ptdw->x_noreio = x;
  ptdw->Tmat = Tmat;

  /** In case of reionization, also calculate the reionized x */
  if(current_ap == ptdw->index_ap_reio){

    /* set x from the evolver (which is very low ~10^-4) as 'xe_before' */
    ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_xe_before] = x;

    /* add the reionization function on top */
    class_call(thermodynamics_reionization_function(z,pth,ptw->ptrp,&x,&dx),
             pth->error_message,
             pth->error_message);

    ptdw->dxdlna = -(1.+z)*dx;
  }

  ptdw->x_reio = x;

  return _SUCCESS_;

}


/**
 * This routine completes for the different codes and approximations
 * the quantity dxdlna for all approximation schemes
 *
 * @param z            Input: redshift
 * @param y            Input: vector of evolver quantities
 * @param dy           Input: vector of evolver derivatives (already filled)
 * @param pth          Input: pointer to thermodynamics structure
 * @param ptw          Input/Output: pointer to thermo workspace
 * @param current_ap   Input: index of the wished approximation scheme
 * @return the error status
 */
int thermodynamics_solve_current_dxdlna(double z,
                                        double * y,
                                        double * dy,
                                        struct thermo * pth,
                                        struct thermo_workspace * ptw,
                                        int current_ap){

  /** Summary: */

  /** Define local variables */
  struct thermo_diffeq_workspace * ptdw = ptw->ptdw;
  struct thermorecfast * precfast = ptw->ptdw->precfast;
  struct thermohyrec * phyrec = ptw->ptdw->phyrec;
  struct thermo_vector * ptv = ptdw->tv;

  /* Thermo quantities */
  double x_H, x_He, x, Tmat;
  /* Analytical quantities */
  double rhs, sqrt_val, drhs_dlna;
  /* Temporary quantities */
  double dx, dx_H_dlna;

  /** Set Tmat from the evolver (it is always evolved). */
  Tmat = y[ptv->index_D_Tmat] + ptw->Tcmb*(1.+z);

  /* Note, dxdlna is always calculated from a steady-state solution of Tmat in 0th order,
   * i.e. approximating dlnTmat_dlna = dlnTrad_dlna = -1. */
  double dlnTmat_dlna = -(1.+z)*(ptv->dy[ptv->index_D_Tmat] + ptw->Tcmb)/Tmat;//-1.; /* Steady state approximation to 0th order */

  /** Case RecfastCLASS :: */
  if(pth->recombination == recfast){
    /** - --> first regime: H and Helium fully ionized */
    /* Already set */
    /** - --> second regime: first Helium recombination (analytic approximation) */
    /* Already set */
    /** - --> third regime: first Helium recombination finished */
    /* Already set */
    /** - --> fourth regime: second Helium recombination starts */
    /* Already set */
    /** - --> fifth regime: Hydrogen recombination starts */
    if (current_ap == ptdw->index_ap_H) {
      /* analytic approximations */
      rhs = exp(1.5*log(precfast->CR*Tmat/(1.+z)/(1.+z)) - precfast->CB1/Tmat)/ptw->SIunit_nH0;
      sqrt_val = sqrt(pow(rhs,2)+4.*rhs);
      drhs_dlna = rhs*((precfast->CB1*dlnTmat_dlna/ptdw->Tmat)+1.5*(dlnTmat_dlna+2.) );
      dx_H_dlna = 0.5*(  (rhs + 2.)/sqrt_val   -   1.  )*drhs_dlna;
      ptdw->dxdlna = dx_H_dlna + ptw->fHe * (1.+z)*dy[ptv->index_x_He];
    }
    /** - --> sixth regime: full recombination */
    else if (current_ap == ptdw->index_ap_frec) {
      ptdw->dxdlna = (1.+z)*(dy[ptv->index_x_H] + ptw->fHe * dy[ptv->index_x_He]);
    }
    /** - --> sixth regime: reionization */
    else if (current_ap == ptdw->index_ap_reio) {
      /* Add the part from background evolution */
      //printf("Contributions : evolv = %.30e , reio = %.30e \n",-(1.+z)*(dy[ptv->index_x_H] + ptw->fHe * dy[ptv->index_x_He]),ptdw->dxdlna);
      ptdw->dxdlna += (1.+z)*(dy[ptv->index_x_H] + ptw->fHe * dy[ptv->index_x_He]);
    }
  }
  /** Case HyRec :: */
  /* Already set */

  return _SUCCESS_;
}

/**
 * This routine computes the quantities connected to interacting dark
 * matter with baryons, idm_b. DCH
 *
 * @param z            Input: redshift
 * @param y            Input: vector of evolver quantities
 * @param pth          Input: pointer to thermodynamics structure
 * @param ptw          Input/Output: pointer to thermo workspace
 * @param pvecback     Input: vector of background quantities
 *
 * @return the error status
 *
 */
int thermodynamics_solve_current_idm_b(struct background * pba,
                                       double z,
                                       double * dy,
                                       double * y,
                                       struct thermo * pth,
                                       struct thermo_workspace * ptw,
                                       double * pvecback){
  /** Summary: */

  /** Define local variables */
  struct thermo_diffeq_workspace * ptdw = ptw->ptdw;
  struct thermorecfast * precfast = ptw->ptdw->precfast;
  struct thermohyrec * phyrec = ptw->ptdw->phyrec;
  struct thermo_vector * ptv = ptdw->tv;

  /* Thermo quantities */
  double Tmat, T_idm_b, Vrms_idm_b2, m_b, T_diff_idm_b, fHe;

  /** Set Tmat from the evolver (it is always evolved). */
  Tmat = y[ptv->index_D_Tmat] + ptw->Tcmb*(1.+z);
  T_idm_b = y[ptv->index_T_idm_b];

  /* dark matter bulk velocity !! Needs better treatment */
  if (z > 1.e3)
    Vrms_idm_b2 = 1.e-8;
  else
    Vrms_idm_b2 = 1.e-8*pow(((1.+z)/1.e3),2);

  fHe = 1- ptw->YHe; //DCH check this
  m_b = _m_p_*_c_*_c_/_eV_; // For now we always assume scattering with protons. This will be adapted in future versions

  T_diff_idm_b = (Tmat*_k_B_/_eV_/m_b)+(T_idm_b*_k_B_/_eV_/pth->m_idm)+(Vrms_idm_b2/3.0); // Get everything (m and T) in eV

  ptdw->R_idm_b = (pvecback[pba->index_bg_a]*pvecback[pba->index_bg_rho_b]*pth->cross_idm_b*pth->n_coeff_idm_b/(m_b+pth->m_idm))
    *powf(T_diff_idm_b,(pth->n_index_idm_b+1.0)/2.0)*fHe
    *(3.e-4*powf(_c_,4.)/(8.*_PI_*_Mpc_over_m_*_G_*_eV_)); //conversion coefficient for the units, might need some checking DCH

  // derivative of R_idm_b wrt to redshift. In perturbations, we will take this wrt *conformal time*
  ptdw->R_idm_b_prime = (pvecback[pba->index_bg_rho_b]*pth->cross_idm_b*pth->n_coeff_idm_b*fHe/(m_b+pth->m_idm))
    *powf(T_diff_idm_b,((pth->n_index_idm_b-1.0)/2.0))
    *(-pvecback[pba->index_bg_a]*pvecback[pba->index_bg_a]*T_diff_idm_b
      +pvecback[pba->index_bg_a]*((pth->n_index_idm_b+1.0)/2.0)*((dy[ptv->index_D_Tmat]+ptw->Tcmb)/m_b + dy[ptv->index_T_idm_b]/pth->m_idm))
    *(3.e-4*powf(_c_,4.)/(8.*_PI_*_Mpc_over_m_*_G_*_eV_)); //conversion coefficient for the units, might need some checking DCH

  ptdw->c_idm_b2 = T_idm_b/pth->m_idm*_eV_/(_c_*_c_);

  /* Store all relevant quantities */
  ptdw->Tmat = Tmat;
  ptdw->T_idm_b = T_idm_b;

  // check fHe vs Yhe TODO DCH

 return _SUCCESS_;

}


/**
 * Initialize the field '->tv' of a thermo_diffeq_workspace structure, which is a thermo_vector structure. This structure contains indices
 * and values of all quantities which need to be integrated with respect to time (and only them: quantities fixed analytically or obeying
 * constraint equations are NOT included in this vector).
 *
 * The routine sets and allocates the vector y, dy and used_in_output with the right size depending on the current approximation scheme
 * stored in the workspace. Moreover the initial conditions for each approximation scheme are calculated and set correctly.
 *
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param mz         Input: negative redshift
 * @param ptw        Input/Output: pointer to thermodynamics workspace
 *
 * @return the error status
 */
int thermodynamics_vector_init(struct precision * ppr,
                               struct background * pba,
                               struct thermo * pth,
                               double mz,
                               struct thermo_workspace * ptw){

  /** Summary: */

  /** Define local variables */
  int index_tv;
  /* ptdw->tv unallocated if ap_current == index_ap_brec, allocated and filled otherwise */
  struct thermo_vector * ptv;
  struct thermo_diffeq_workspace * ptdw = ptw->ptdw;

  double z,x,Tmat;
  double T_idm_b; //DCH
  int evolves_xHe = (pth->recombination == recfast);
  int evolves_xH = (pth->recombination == recfast);

  class_alloc(ptv,sizeof(struct thermo_vector),pth->error_message);

  /* mz = Minus z is inverted*/
  z = -mz;

  /* Start from no component */
  index_tv = 0;

  /* Add common indices (Have to be added before) */
  class_define_index(ptv->index_D_Tmat,_TRUE_,index_tv,1);

  if(pth->has_idm_b == _TRUE_){
    class_define_index(ptv->index_T_idm_b,_TRUE_,index_tv,1); //DCH
  }

  /* Add all components that should be evolved */
  if(ptdw->ap_current == ptdw->index_ap_brec){
    /* Nothing else to add */
  }
  else if(ptdw->ap_current == ptdw->index_ap_He1){
    /* Nothing else to add */
  }
  else if(ptdw->ap_current == ptdw->index_ap_He1f){
    /* Nothing else to add */
  }
  else if(ptdw->ap_current == ptdw->index_ap_He2){
    /* Nothing else to add */
  }
  else if(ptdw->ap_current == ptdw->index_ap_H){
    class_define_index(ptv->index_x_He,evolves_xHe,index_tv,1);
  }
  else if(ptdw->ap_current == ptdw->index_ap_frec){
    class_define_index(ptv->index_x_He,evolves_xHe,index_tv,1);
    class_define_index(ptv->index_x_H,evolves_xH,index_tv,1);
  }
  else if(ptdw->ap_current == ptdw->index_ap_reio){
    class_define_index(ptv->index_x_He,evolves_xHe,index_tv,1);
    class_define_index(ptv->index_x_H,evolves_xH,index_tv,1);
  }

  /* We have now obtained the full size */
  ptv->tv_size = index_tv;

  /* Allocate all arrays used during the evolution */
  class_calloc(ptv->y,ptv->tv_size,sizeof(double),pth->error_message);
  class_alloc(ptv->dy,ptv->tv_size*sizeof(double),pth->error_message);
  class_alloc(ptv->used_in_output,ptv->tv_size*sizeof(int),pth->error_message);

  for (index_tv=0; index_tv<ptv->tv_size; index_tv++){
    ptv->used_in_output[index_tv] = _TRUE_;
  }

  /* setting intial conditions for each approximation: */

  /** - HyRec */
  if(pth->recombination == hyrec){
    /* Initial initialization */
    if(ptdw->ap_current == ptdw->index_ap_brec){
      ptdw->tv = ptv;
      ptdw->tv->y[ptdw->tv->index_D_Tmat] = 0.;
      ptdw->tv->dy[ptdw->tv->index_D_Tmat] = 0.;
    }
    /* Afterwards initialization */
    else{
      /* Free the old vector and its indices */
      ptv->y[ptv->index_D_Tmat] = ptdw->tv->y[ptdw->tv->index_D_Tmat];
      ptv->dy[ptv->index_D_Tmat] = ptdw->tv->dy[ptdw->tv->index_D_Tmat];
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);
      ptdw->tv = ptv;
    }
  }

  /** RecfastCLASS */
  if(pth->recombination == recfast){
    /* - in the first scheme (brec = before recombination), we need initial condition for the matter temperature given by the photon temperature */
    if(ptdw->ap_current == ptdw->index_ap_brec){
      /* Store Tmat in workspace for later use */
      ptdw->Tmat = ptw->Tcmb*(1.+z);
      ptdw->dTmat = ptw->Tcmb;

      /* Set the new vector and its indices */
      ptdw->tv = ptv;

      ptdw->tv->y[ptdw->tv->index_D_Tmat] = 0.;
      ptdw->tv->dy[ptdw->tv->index_D_Tmat] = 0.;

      ptdw->require_H = _FALSE_;
      ptdw->require_He = _FALSE_;
    }
    /* - in this scheme we start to evolve Helium and thus need to set its initial condition via the analytic function */
    else if(ptdw->ap_current == ptdw->index_ap_H){
      /* Store Tmat in workspace for later use */
      ptdw->Tmat = ptdw->tv->y[ptdw->tv->index_D_Tmat] + ptw->Tcmb*(1.+z);
      ptdw->dTmat = -ptdw->tv->dy[ptdw->tv->index_D_Tmat] + ptw->Tcmb;

      /* Obtain initial contents of new vector analytically, especially x_He */
      class_call(thermodynamics_solve_current_quantities(z,ptdw->tv->y,pth,ptw,ptdw->ap_current-1),
                 pth->error_message,
                 pth->error_message);

      /* Set the new vector and its indices */
      ptv->y[ptv->index_D_Tmat] = ptdw->tv->y[ptdw->tv->index_D_Tmat];
      ptv->dy[ptv->index_D_Tmat] = ptdw->tv->dy[ptdw->tv->index_D_Tmat];
      ptv->y[ptv->index_x_He] = ptdw->x_He;
      ptv->dy[ptv->index_x_He] = -ptdw->dx_He;

      /* Free the old vector and its indices */
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);

      /* Copy the new vector into the position of the old one*/
      ptdw->tv = ptv;

      ptdw->require_H = _FALSE_;
      ptdw->require_He = _TRUE_;
    }
    /* - in the scheme of full recombination (=frec) we evolve all quantities and thus need to set their initial conditions.
         Tmat and x_He are solely taken from the previous scheme, x_H is set via the analytic function */
    else if(ptdw->ap_current == ptdw->index_ap_frec){
      /* Store Tmat in workspace for later use */
      ptdw->Tmat = ptdw->tv->y[ptdw->tv->index_D_Tmat] + ptw->Tcmb*(1.+z);
      ptdw->dTmat = -ptdw->tv->dy[ptdw->tv->index_D_Tmat] + ptw->Tcmb;

      /* Obtain initial contents of new vector analytically, especially x_H */
      class_call(thermodynamics_solve_current_quantities(z,ptdw->tv->y,pth,ptw,ptdw->ap_current-1),
                 pth->error_message,
                 pth->error_message);

      /* Set the new vector and its indices */
      ptv->y[ptv->index_D_Tmat] = ptdw->tv->y[ptdw->tv->index_D_Tmat];
      ptv->dy[ptv->index_D_Tmat] = ptdw->tv->dy[ptdw->tv->index_D_Tmat];
      ptv->y[ptv->index_x_H] = ptdw->x_H;
      ptv->dy[ptv->index_x_H] = -ptdw->dx_H;
      ptv->y[ptv->index_x_He] = ptdw->tv->y[ptdw->tv->index_x_He];
      ptv->dy[ptv->index_x_He] = ptdw->tv->dy[ptdw->tv->index_x_He];

      /* Free the old vector and its indices */
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);

      /* Copy the new vector into the position of the old one*/

      ptdw->tv = ptv;

      ptdw->require_H = _TRUE_;
      ptdw->require_He = _TRUE_;
    }
    /* - during reionization we continue to evolve all quantities. Now all three intial conditions are just taken from the previous scheme */
    else if(ptdw->ap_current == ptdw->index_ap_reio){

      /* Set the new vector and its indices */
      ptv->y[ptv->index_D_Tmat] = ptdw->tv->y[ptdw->tv->index_D_Tmat];
      ptv->dy[ptv->index_D_Tmat] = ptdw->tv->dy[ptdw->tv->index_D_Tmat];
      ptv->y[ptv->index_x_H] = ptdw->tv->y[ptdw->tv->index_x_H];
      ptv->dy[ptv->index_x_H] = ptdw->tv->dy[ptdw->tv->index_x_H];
      ptv->y[ptv->index_x_He] = ptdw->tv->y[ptdw->tv->index_x_He];
      ptv->dy[ptv->index_x_He] = ptdw->tv->dy[ptdw->tv->index_x_He];

      /* Free the old vector and its indices */
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);

      /* Copy the new vector into the position of the old one*/

      ptdw->tv = ptv;

      ptdw->require_H = _TRUE_;
      ptdw->require_He = _TRUE_;
    }
    /* - in all other approximations we only evolve Tmat and set its initial conditions from the previous scheme */
    else{
      /* Store Tmat in workspace for later use */
      ptdw->Tmat = ptdw->tv->y[ptdw->tv->index_D_Tmat] + ptw->Tcmb*(1.+z);
      ptdw->dTmat = -ptdw->tv->dy[ptdw->tv->index_D_Tmat] + ptw->Tcmb;

      /* Set the new vector and its indices */
      ptv->y[ptv->index_D_Tmat] = ptdw->tv->y[ptdw->tv->index_D_Tmat];
      ptv->dy[ptv->index_D_Tmat] = ptdw->tv->dy[ptdw->tv->index_D_Tmat];
      /* Free the old vector and its indices */
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);

      /* Copy the new vector into the position of the old one*/
      ptdw->tv = ptv;

      ptdw->require_H = _FALSE_;
      ptdw->require_He = _FALSE_;
    }
  }

  //  DCH initialise T_idm_b*/
  if(pth->has_idm_b == _TRUE_){

    if(ptdw->ap_current == ptdw->index_ap_brec){

      if(pth->n_index_idm_b == -4){ //special treatment for this case, in which the baryons and DM are not tightly coupled at early times
        ptdw->T_idm_b = 0.;
        ptdw->dT_idm_b = 0.;
      }
      else{
        ptdw->T_idm_b = ptw->Tcmb*(1.+z);
        ptdw->dT_idm_b = ptw->Tcmb;
      }

      /* Set the new vector and its indices */
      ptdw->tv = ptv;
      ptdw->tv->y[ptdw->tv->index_T_idm_b] = 0.;
      ptdw->tv->dy[ptdw->tv->index_T_idm_b] = 0.;
    }

    /* Afterwards initialization */
    else{
      /* Free the old vector and its indices */
      ptv->y[ptv->index_T_idm_b] = ptdw->tv->y[ptdw->tv->index_T_idm_b];
      ptv->dy[ptv->index_T_idm_b] = ptdw->tv->dy[ptdw->tv->index_T_idm_b];
      class_call(thermodynamics_vector_free(ptdw->tv),
                 pth->error_message,
                 pth->error_message);
      ptdw->tv = ptv;
    }
  }

  return _SUCCESS_;
}


/**
 * Free the thermo_vector structure, which is the '->tv' field of the thermodynamics_differential_workspace ptdw structure
 *
 * @param tv        Input: pointer to thermo_vector structure to be freed
 * @return the error status
 */
int thermodynamics_vector_free(struct thermo_vector * tv){

  free(tv->y);
  free(tv->dy);
  free(tv->used_in_output);
  free(tv);

  return _SUCCESS_;

}


/**
 * Initialize the thermodynamics workspace.
 * It contains the workspace for solving differential equations (dubbed thermo_diffeq_workspace). In there, all approximations are stored
 * It contains the reionization parameters struct In there, all parameters relating to analytical reionization are stored
 * It contains the heating parameters struct  In there, all parameters related to heating are stored
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param ptw        Input/Output: pointer to thermodynamics workspace
 * @return the error status
 */
int thermodynamics_workspace_init(struct precision * ppr,
                                  struct background * pba,
                                  struct thermo * pth,
                                  struct thermo_workspace * ptw){

  /** Summary: */

  /** Define local variables */
  int index_ap;

  /** Allocate the workspace used henceforth to store all temporary quantities */
  class_alloc(ptw->ptdw,
              sizeof(struct thermo_diffeq_workspace),
              pth->error_message);
  class_alloc(ptw->ptrp,
              sizeof(struct thermo_reionization_parameters),
              pth->error_message);

  /** - Start with thermo_diffeq_workspace */

  /**   - count number of approximations, initialize their indices */
  index_ap=0;

  ptw->Nz_reco_lin = ppr->thermo_Nz_lin;
  ptw->Nz_reco_log = ppr->thermo_Nz_log;
  ptw->Nz_reio = ppr->reionization_z_start_max / ppr->reionization_sampling;
  ptw->Nz_reco = ptw->Nz_reco_lin + ptw->Nz_reco_log;
  ptw->Nz_tot = ptw->Nz_reio + ptw->Nz_reco;

  /* YHe, fHe*/
  ptw->YHe = pth->YHe;
  ptw->fHe = pth->fHe;

  /* Tnow */
  ptw->Tcmb = pba->T_cmb;

  /* H0 in inverse seconds (while pba->H0 is [H0/c] in inverse Mpcs) */
  ptw->SIunit_H0 = pba->H0 * _c_ / _Mpc_over_m_;

  /* Number of hydrogen nuclei today in m**-3 */
  ptw->SIunit_nH0 = 3.*ptw->SIunit_H0*ptw->SIunit_H0*pba->Omega0_b/(8.*_PI_*_G_*_m_H_)*(1.-ptw->YHe);
  pth->n_e = ptw->SIunit_nH0;
  ptw->x_limit_T =  ppr->recfast_H_frac;

  //TODO :: is this necessary?
  ptw->ptdw->x_reio = 1.+2.*ptw->fHe;
  ptw->ptdw->x_noreio = 1.+2.*ptw->fHe;

  /** - define approximations */
  /* Approximations have to appear in chronological order here! */
  class_define_index(ptw->ptdw->index_ap_brec,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_He1,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_He1f,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_He2,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_H,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_frec,_TRUE_,index_ap,1);
  class_define_index(ptw->ptdw->index_ap_reio,_TRUE_,index_ap,1);
  ptw->ptdw->ap_size=index_ap;

  /** - store all ending redshifts for each approximation */
  class_alloc(ptw->ptdw->ap_z_limits,ptw->ptdw->ap_size*sizeof(double),pth->error_message);

  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_brec] = ppr->recfast_z_He_1+ppr->recfast_delta_z_He_1;
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_He1] = ppr->recfast_z_He_2+ppr->recfast_delta_z_He_2;
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_He1f] = ppr->recfast_z_He_3+ppr->recfast_delta_z_He_3;
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_He2] = 2870.;// TODO :: set correctly
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_H] = 1600.;// TODO :: set correctly
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_frec] = ppr->reionization_z_start_max;
  ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_reio] = 0.0;

  /** - store smoothing deltas for transitions at the beginning of each aproximation */
  class_alloc(ptw->ptdw->ap_z_limits_delta,ptw->ptdw->ap_size*sizeof(double),pth->error_message);

  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_brec] = 0.;
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_He1] = ppr->recfast_delta_z_He_1;
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_He1f] = ppr->recfast_delta_z_He_2;
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_He2] = ppr->recfast_delta_z_He_3;
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_H] = 50.; // TODO :: set correctly
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_frec] = 50.;// TODO :: set correctly
  ptw->ptdw->ap_z_limits_delta[ptw->ptdw->index_ap_reio] = 2.0;// TODO :: set correctly

  /*fix current approximation scheme to before recombination */
  ptw->ptdw->ap_current = ptw->ptdw->index_ap_brec;

  /* With recombination computed by HyRec or Recfast, we need to initialize the wrappers */
  if(pth->recombination == hyrec){
    class_alloc(ptw->ptdw->phyrec,
                sizeof(struct thermohyrec),
                pth->error_message);

    ptw->ptdw->phyrec->thermohyrec_verbose = pth->hyrec_verbose;

    class_call(thermodynamics_hyrec_init(ppr,ptw->SIunit_nH0,pba->T_cmb,ptw->fHe, ptw->ptdw->ap_z_limits[ptw->ptdw->index_ap_brec],ptw->ptdw->phyrec),
               ptw->ptdw->phyrec->error_message,
               pth->error_message);
  }
  else if(pth->recombination == recfast){
    class_alloc(ptw->ptdw->precfast,
                sizeof(struct thermorecfast),
                pth->error_message);

    class_call(thermodynamics_recfast_init(ppr,pba,pth,ptw->ptdw->precfast,ptw->fHe),
               ptw->ptdw->precfast->error_message,
               pth->error_message);
  }

  return _SUCCESS_;

}


/**
 * Free the thermo_workspace structure (with the exception of the thermo_vector '->tv' field, which is freed separately in
 * thermo_vector_free).
 *
 * @param ptw        Input: pointer to perturb_workspace structure to be freed
 * @return the error status
 */
int thermodynamics_workspace_free(struct thermo* pth,
                                  struct thermo_workspace * ptw) {

  free(ptw->ptdw->ap_z_limits);
  free(ptw->ptdw->ap_z_limits_delta);

  if(pth->recombination == hyrec){
    class_call(thermodynamics_hyrec_free(ptw->ptdw->phyrec),
               ptw->ptdw->phyrec->error_message,
               pth->error_message);
    free(ptw->ptdw->phyrec);
  }
  else if(pth->recombination == recfast){
    free(ptw->ptdw->precfast);
  }

  free(ptw->ptrp->reionization_parameters);
  free(ptw->ptdw);
  free(ptw->ptrp);

  free(ptw);

  return _SUCCESS_;

}


/**
 * This routine initializes reionization_parameters for the chosen scheme of reionization function.
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param preio      Input/Output: pointer to the reionization parameters structure
 * @return the error status
 */
int thermodynamics_set_parameters_reionization(struct precision * ppr,
                                               struct background * pba,
                                               struct thermo * pth,
                                               struct thermo_reionization_parameters * preio){

  /** Summary: */

  /** Define local variables */
  int bin;
  int point;
  double xe_input,xe_actual,z_sup;

  /** - allocate the vector of parameters defining the function \f$ X_e(z) \f$ */
  class_alloc(preio->reionization_parameters,preio->reio_num_params*sizeof(double),pth->error_message);


  class_test(ppr->reionization_sampling <= 0.0,
             pth->error_message,
             "stop to avoid division by zero. Reionization stepsize has to be larger than zero");

  /** - (a) if reionization implemented like in CAMB */

  if ((pth->reio_parametrization == reio_camb) || (pth->reio_parametrization == reio_half_tanh)) {

    /** - --> set values of these parameters, excepted those depending on the reionization redshift */

    if (pth->reio_parametrization == reio_camb) {
      preio->reionization_parameters[preio->index_reio_xe_after] = 1. + pth->YHe/(_not4_*(1.-pth->YHe));  /* xe_after_reio: H + singly ionized He (note: segmentation fault impossible,
                                                                                                             checked before that denominator is non-zero) */
    }
    if (pth->reio_parametrization == reio_half_tanh) {
      preio->reionization_parameters[preio->index_reio_xe_after] = 1.; /* xe_after_reio: neglect He ionization */
      //+ 2*pth->YHe/(_not4_*(1.-pth->YHe));    /* xe_after_reio: H + fully ionized He */
    }

    preio->reionization_parameters[preio->index_reio_exponent] = pth->reionization_exponent; /* reio_exponent */
    preio->reionization_parameters[preio->index_reio_width] = pth->reionization_width;    /* reio_width */
    preio->reionization_parameters[preio->index_helium_fullreio_fraction] = pth->YHe/(_not4_*(1.-pth->YHe)); /* helium_fullreio_fraction (note: segmentation fault impossible,
                                                                                                                checked before that denominator is non-zero) */
    preio->reionization_parameters[preio->index_helium_fullreio_redshift] = pth->helium_fullreio_redshift; /* helium_fullreio_redshift */
    preio->reionization_parameters[preio->index_helium_fullreio_width] = pth->helium_fullreio_width;    /* helium_fullreio_width */

    class_test(preio->reionization_parameters[preio->index_reio_exponent]==0,
               pth->error_message,
               "stop to avoid division by zero");

    class_test(preio->reionization_parameters[preio->index_reio_width]==0,
               pth->error_message,
               "stop to avoid division by zero");

    class_test(preio->reionization_parameters[preio->index_helium_fullreio_width]==0,
               pth->error_message,
               "stop to avoid division by zero");

    /** - --> if reionization redshift given as an input, initialize the remaining values*/

    if (pth->reio_z_or_tau == reio_z) {

      /* reionization redshift */
      preio->reionization_parameters[preio->index_reio_redshift] = pth->z_reio;

      /* infer starting redshift for hydrogen */

      if (pth->reio_parametrization == reio_camb) {

        preio->reionization_parameters[preio->index_reio_start] = preio->reionization_parameters[preio->index_reio_redshift]+
                                                                  ppr->reionization_start_factor*pth->reionization_width;

        /* if starting redshift for helium is larger, take that one (does not happen in realistic models) */
        if (preio->reionization_parameters[preio->index_reio_start] <
            pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width)

          preio->reionization_parameters[preio->index_reio_start] =
            pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width;

      }
      else {

        preio->reionization_parameters[preio->index_reio_start] = pth->z_reio;
      }

      class_test(preio->reionization_parameters[preio->index_reio_start] > ppr->reionization_z_start_max,
                 pth->error_message,
                 "starting redshift for reionization > reionization_z_start_max = %e\n",ppr->reionization_z_start_max);

    }

    /** - --> if reionization optical depth given as an input, find reionization redshift by dichotomy and initialize the remaining values */
    if (pth->reio_z_or_tau == reio_tau) {
           z_sup = ppr->reionization_z_start_max-ppr->reionization_start_factor*pth->reionization_width;
      class_test(z_sup < 0.,
                 pth->error_message,
                 "parameters are such that reionization cannot take place before today while starting after z_start_max; need to increase z_start_max");

      /* maximum possible reionization redshift */
      preio->reionization_parameters[preio->index_reio_redshift] = z_sup;
      /* maximum possible starting redshift */
      preio->reionization_parameters[preio->index_reio_start] = ppr->reionization_z_start_max;
    }

  }
  /** - (b) if reionization implemented with reio_bins_tanh scheme */

  else if (pth->reio_parametrization == reio_bins_tanh) {

    /* this algorithm requires at least two bin centers (i.e. at least 4 values in the (z,xe) array, counting the edges). */
    class_test(pth->binned_reio_num<2,
               pth->error_message,
               "current implementation of binned reio requires at least two bin centers");

    /* check that this input can be interpreted by the code */
    for (bin=1; bin<pth->binned_reio_num; bin++) {
      class_test(pth->binned_reio_z[bin-1]>=pth->binned_reio_z[bin],
                 pth->error_message,
                 "value of reionization bin centers z_i expected to be passed in growing order: %e, %e",
                 pth->binned_reio_z[bin-1],
                 pth->binned_reio_z[bin]);
    }

    /* the code will not only copy here the "bin centers" passed in input. It will add an initial and final value for (z,xe).
       First, fill all entries except the first and the last */
    for (bin=1; bin<preio->reio_num_z-1; bin++) {
      preio->reionization_parameters[preio->index_reio_first_z+bin] = pth->binned_reio_z[bin-1];
      preio->reionization_parameters[preio->index_reio_first_xe+bin] = pth->binned_reio_xe[bin-1];
    }

    /* find largest value of z in the array. We choose to define it as z_(i_max) + 2*(the distance between z_(i_max) and z_(i_max-1)). E.g. if
       the bins are in 10,12,14, the largest z will be 18. */
    preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1] =
      preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-2]
      +2.*(preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-2]
        -preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-3]);

    /* copy this value in reio_start */
    preio->reionization_parameters[preio->index_reio_start] = preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1];

    /* check it's not too big */
    class_test(preio->reionization_parameters[preio->index_reio_start] > ppr->reionization_z_start_max,
               pth->error_message,
               "starting redshift for reionization = %e, reionization_z_start_max = %e, you must change the binning or increase reionization_z_start_max",
               preio->reionization_parameters[preio->index_reio_start],
               ppr->reionization_z_start_max);

    /* find smallest value of z in the array. We choose to define it as z_0 - (the distance between z_1 and z_0). E.g. if
       the bins are in 10,12,14, the stop redshift will be 8. */
    preio->reionization_parameters[preio->index_reio_first_z] =
      2.*preio->reionization_parameters[preio->index_reio_first_z+1]
      -preio->reionization_parameters[preio->index_reio_first_z+2];

    /* check it's not too small */
    /* 6.06.2015: changed this test to simply imposing that the first z is at least zero */
    /*
    class_test(preio->reionization_parameters[preio->index_reio_first_z] < 0,
               pth->error_message,
               "final redshift for reionization = %e, you must change the binning or redefine the way in which the code extrapolates below the first value of z_i",preio->reionization_parameters[preio->index_reio_first_z]);
    */
    if (preio->reionization_parameters[preio->index_reio_first_z] < 0) {
      preio->reionization_parameters[preio->index_reio_first_z] = 0.;
    }

    /* infer xe after reio */
    preio->reionization_parameters[preio->index_reio_first_xe] = 1. + pth->YHe/(_not4_*(1.-pth->YHe)); /* xe_after_reio: H + singly ionized He (note: segmentation fault impossible,
                                                                                                          checked before that denominator is non-zero) */

    /* pass step sharpness parameter */
    preio->reionization_parameters[preio->index_reio_step_sharpness] = pth->binned_reio_step_sharpness;

  }

  /** - (c) if reionization implemented with reio_many_tanh scheme */
  else if (pth->reio_parametrization == reio_many_tanh) {

    /* this algorithm requires at least one jump centers */
    class_test(pth->many_tanh_num<1,
               pth->error_message,
               "current implementation of reio_many_tanh requires at least one jump center");

    /* check that z input can be interpreted by the code */
    for (bin=1; bin<pth->many_tanh_num; bin++) {
      class_test(pth->many_tanh_z[bin-1]>=pth->many_tanh_z[bin],
                 pth->error_message,
                 "value of reionization bin centers z_i expected to be passed in growing order: %e, %e",
                 pth->many_tanh_z[bin-1],
                 pth->many_tanh_z[bin]);

    }

    /* the code will not only copy here the "jump centers" passed in input. It will add an initial and final value for (z,xe).
       First, fill all entries except the first and the last */
    for (bin=1; bin<preio->reio_num_z-1; bin++) {

      preio->reionization_parameters[preio->index_reio_first_z+bin] = pth->many_tanh_z[bin-1];

      /* check that xe input can be interpreted by the code */
      xe_input = pth->many_tanh_xe[bin-1];
      if (xe_input >= 0.) {
        xe_actual = xe_input;
      }
      //-1 means "after hydrogen + first helium recombination"
      else if ((xe_input<-0.9) && (xe_input>-1.1)) {
        xe_actual = 1. + pth->YHe/(_not4_*(1.-pth->YHe));
      }
      //-2 means "after hydrogen + second helium recombination"
      else if ((xe_input<-1.9) && (xe_input>-2.1)) {
        xe_actual = 1. + 2.*pth->YHe/(_not4_*(1.-pth->YHe));
      }
      //other negative number is nonsense
      else {
        class_stop(pth->error_message,
                   "Your entry for many_tanh_xe[%d] is %e, this makes no sense (either positive or 0,-1,-2)",
                   bin-1,pth->many_tanh_xe[bin-1]);
      }

      preio->reionization_parameters[preio->index_reio_first_xe+bin] = xe_actual;
    }

    /* find largest value of z in the array. We choose to define it as z_(i_max) + ppr->reionization_start_factor*step_sharpness. */
    preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1] =
      preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-2]
      +ppr->reionization_start_factor*pth->many_tanh_width;

    /* copy this value in reio_start */
    preio->reionization_parameters[preio->index_reio_start] = preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1];

    /* check it's not too big */
    class_test(preio->reionization_parameters[preio->index_reio_start] > ppr->reionization_z_start_max,
               pth->error_message,
               "starting redshift for reionization = %e, reionization_z_start_max = %e, you must change the binning or increase reionization_z_start_max",
               preio->reionization_parameters[preio->index_reio_start],
               ppr->reionization_z_start_max);

    /* find smallest value of z in the array. We choose to define it as z_0 - ppr->reionization_start_factor*step_sharpness, but at least zero. */
    preio->reionization_parameters[preio->index_reio_first_z] =
      preio->reionization_parameters[preio->index_reio_first_z+1]
      -ppr->reionization_start_factor*pth->many_tanh_width;

    if (preio->reionization_parameters[preio->index_reio_first_z] < 0) {
      preio->reionization_parameters[preio->index_reio_first_z] = 0.;
    }

    /* infer xe after reio */
    preio->reionization_parameters[preio->index_reio_first_xe] = preio->reionization_parameters[preio->index_reio_first_xe+1];

    /* if we want to model only hydrogen reionization and neglect both helium reionization */
    //preio->reionization_parameters[preio->index_reio_first_xe] = 1.;

    /* if we want to model only hydrogen + first helium reionization and neglect second helium reionization */
    //preio->reionization_parameters[preio->index_reio_first_xe] = 1. + pth->YHe/(_not4_*(1.-pth->YHe));

    /* if we want to model hydrogen + two helium reionization */
    //preio->reionization_parameters[preio->index_reio_first_xe] = 1. + 2.*pth->YHe/(_not4_*(1.-pth->YHe));

    /* pass step sharpness parameter */
    class_test(pth->many_tanh_width<=0,
               pth->error_message,
               "many_tanh_width must be strictly positive, you passed %e",
               pth->many_tanh_width);

    preio->reionization_parameters[preio->index_reio_step_sharpness] = pth->many_tanh_width;

  }

  /** - (d) if reionization implemented with reio_inter scheme */
  else if (pth->reio_parametrization == reio_inter) {

    /* this parametrization requires at least one point (z,xe) */
    class_test(pth->reio_inter_num<1,
               pth->error_message,
               "current implementation of reio_inter requires at least one point (z,xe)");

    /* this parametrization requires that the first z value is zero */
    class_test(pth->reio_inter_z[0] != 0.,
               pth->error_message,
               "For reio_inter scheme, the first value of reio_inter_z[...]  should always be zero, you passed %e",
               pth->reio_inter_z[0]);

    /* check that z input can be interpreted by the code */
    for (point=1; point<pth->reio_inter_num; point++) {
      class_test(pth->reio_inter_z[point-1]>=pth->reio_inter_z[point],
                 pth->error_message,
                 "value of reionization bin centers z_i expected to be passed in growing order, unlike: %e, %e",
                 pth->reio_inter_z[point-1],
                 pth->reio_inter_z[point]);
    }

    /* this parametrization requires that the last x_i value is zero (the code will substitute it with the value that one would get in
       absence of reionization, as compute by the recombination code) */
    class_test(pth->reio_inter_xe[pth->reio_inter_num-1] != 0.,
               pth->error_message,
               "For reio_inter scheme, the last value of reio_inter_xe[...]  should always be zero, you passed %e",
               pth->reio_inter_xe[pth->reio_inter_num-1]);

    /* copy here the (z,xe) values passed in input. */
    for (point=0; point<preio->reio_num_z; point++) {

      preio->reionization_parameters[preio->index_reio_first_z+point] = pth->reio_inter_z[point];

      /* check that xe input can be interpreted by the code */
      xe_input = pth->reio_inter_xe[point];
      if (xe_input >= 0.) {
        xe_actual = xe_input;
      }
      //-1 means "after hydrogen + first helium recombination"
      else if ((xe_input<-0.9) && (xe_input>-1.1)) {
        xe_actual = 1. + pth->YHe/(_not4_*(1.-pth->YHe));
      }
      //-2 means "after hydrogen + second helium recombination"
      else if ((xe_input<-1.9) && (xe_input>-2.1)) {
        xe_actual = 1. + 2.*pth->YHe/(_not4_*(1.-pth->YHe));
      }
      //other negative number is nonsense
      else {
        class_stop(pth->error_message,
                   "Your entry for reio_inter_xe[%d] is %e, this makes no sense (either positive or 0,-1,-2)",
                   point,pth->reio_inter_xe[point]);
      }

      preio->reionization_parameters[preio->index_reio_first_xe+point] = xe_actual;
    }

    /* copy highest redshift in reio_start */
    preio->reionization_parameters[preio->index_reio_start] = preio->reionization_parameters[preio->index_reio_first_z+preio->reio_num_z-1];

    /* check it's not too big */
    class_test(preio->reionization_parameters[preio->index_reio_start] > ppr->reionization_z_start_max,
               pth->error_message,
               "starting redshift for reionization = %e, reionization_z_start_max = %e, you must change the binning or increase reionization_z_start_max",
               preio->reionization_parameters[preio->index_reio_start],
               ppr->reionization_z_start_max);

  }
  else if (pth->reio_parametrization == reio_none) {
    printf("Computing thermodynamics without reionization \n");
  }
  else{
    class_stop(pth->error_message,
               "Other reionization schemes not supported yet.");
  }

 return _SUCCESS_;

}


/**
 * This function is called in the evolvers loop if the input for reionization is tau_reio. Instead of the final evolution in the last
 * iteration, reionization is computed by this function instead. Instead of computing for a fixed z_reio, we find z_reio by bisection.
 * First we make an initial guess for z_reio with reionization_z_start_max and then find a z_reio which leads to the given tau_reio
 * (in the range of tolerance reionization_optical_depth_tol).
 *
 * @param ptpaw      Input: pointer to parameters and workspace
 * @param mz_ini     Input: initial redshift
 * @param mz_end     Input: ending redshift
 * @param mz_output  Input: pointer to redshift array
 * @param Nz         Input: number of redshift values in array
 * @return the error status
 */
int thermodynamics_reionization_evolve_with_tau(struct thermodynamics_parameters_and_workspace * ptpaw,
                                                double mz_ini,
                                                double mz_end,
                                                double * mz_output,
                                                int Nz){

  /** Summary: */

  /** Define local variables */
  int counter;
  double z_sup,z_mid,z_inf;
  double tau_sup,tau_mid,tau_inf;

  int index_tv;

  struct precision * ppr;
  struct background * pba;
  struct thermo * pth;
  struct thermo_workspace * ptw;

  /* function pointer to ODE evolver and names of possible evolvers */
  extern int evolver_rk();
  extern int evolver_ndf15();
  int (*generic_evolver)();

  /** - Remame fields to avoid heavy notations */
  ppr = ptpaw->ppr;
  pba = ptpaw->pba;
  pth = ptpaw->pth;
  ptw = ptpaw->ptw;

  /** - Initialize two thermo vectors, one to store initial conditions and one temporary vector for the calculations in the bisection */
  struct thermo_vector * ptv; // Temporary vector as workspace
  struct thermo_vector * ptvs; // Vector for storing the initial conditions
  ptvs = ptw->ptdw->tv;

  int evolves_xHe = (pth->recombination == recfast);
  int evolves_xH = (pth->recombination == recfast);


  class_alloc(ptv,sizeof(struct thermo_vector),pth->error_message);

  class_define_index(ptv->index_D_Tmat,_TRUE_,index_tv,1);
  if(pth->has_idm_b == _TRUE_){
    class_define_index(ptv->index_T_idm_b,_TRUE_,index_tv,1); //DCH
  }
  class_define_index(ptv->index_x_He,evolves_xHe,index_tv,1);
  class_define_index(ptv->index_x_H,evolves_xH,index_tv,1);

  /* We have now obtained the full size */
  ptv->tv_size = index_tv;

  /* Allocate all arrays used during the evolution */
  class_calloc(ptv->y,ptv->tv_size,sizeof(double),pth->error_message);
  class_alloc(ptv->dy,ptv->tv_size*sizeof(double),pth->error_message);
  class_alloc(ptv->used_in_output,ptv->tv_size*sizeof(int),pth->error_message);

  for (index_tv=0; index_tv<ptv->tv_size; index_tv++){
    ptv->used_in_output[index_tv] = _TRUE_;
  }

  /** - Assign the temporary vector, then find upper and lower value of bisection */
  ptv->y[ptv->index_D_Tmat] = ptvs->y[ptvs->index_D_Tmat];
  ptv->dy[ptv->index_D_Tmat] = ptvs->dy[ptvs->index_D_Tmat];

  if(pth->has_idm_b == _TRUE_){
    ptv->y[ptv->index_T_idm_b] = ptvs->y[ptvs->index_T_idm_b];//DCH
    ptv->dy[ptv->index_T_idm_b] = ptvs->dy[ptvs->index_T_idm_b];//DCH
  }

  if(evolves_xH){
    ptv->y[ptv->index_x_H] = ptvs->y[ptvs->index_x_H];
    ptv->dy[ptv->index_x_H] = ptvs->dy[ptvs->index_x_H];
  }
  if(evolves_xHe){
    ptv->y[ptv->index_x_He] = ptvs->y[ptvs->index_x_He];
    ptv->dy[ptv->index_x_He] = ptvs->dy[ptvs->index_x_He];
  }

  ptw->ptdw->tv = ptv;

  /* upper value */
  z_sup = ppr->reionization_z_start_max-ppr->reionization_start_factor*pth->reionization_width;
  class_test(z_sup < 0.,
             pth->error_message,
             "parameters are such that reionization cannot take place before today while starting after z_start_max; need to increase z_start_max");

  /* maximum possible reionization redshift */
  ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_redshift] = z_sup;
  /* maximum possible starting redshift */
  ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] = ppr->reionization_z_start_max;

  if(ppr->evolver == rk){
    generic_evolver = evolver_rk;
  }
  else{
    generic_evolver = evolver_ndf15;
  }

  /* Calculate a first ionization history  at upper limit */
  class_call(generic_evolver(thermodynamics_solve_derivs,
                             mz_ini,
                             mz_end,
                             ptv->y,
                             ptv->used_in_output,
                             ptv->tv_size,
                             ptpaw,
                             ppr->tol_thermo_integration,
                             ppr->smallest_allowed_variation,
                             thermodynamics_solve_timescale,  // timescale
                             1., // stepsize
                             mz_output, // values of z for output
                             Nz, // size of previous array
                             thermodynamics_solve_store_sources, // function for output
                             NULL, // print variables
                             pth->error_message),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_reionization_get_tau(ppr,
                                                 pba,
                                                 pth,
                                                 ptw),
             pth->error_message,
             pth->error_message);

  tau_sup=ptw->reionization_optical_depth;

  class_test(tau_sup < pth->tau_reio,
             pth->error_message,
             "parameters are such that reionization cannot start after z_start_max");

  /* Restore initial conditions */
  ptv->y[ptv->index_D_Tmat] = ptvs->y[ptvs->index_D_Tmat];
  ptv->dy[ptv->index_D_Tmat] = ptvs->dy[ptvs->index_D_Tmat];

  if(pth->has_idm_b == _TRUE_){
    ptv->y[ptv->index_T_idm_b] = ptvs->y[ptvs->index_T_idm_b]; //DCH
    ptv->dy[ptv->index_T_idm_b] = ptvs->dy[ptvs->index_T_idm_b]; //DCH
  }

  if(evolves_xH){
    ptv->y[ptv->index_x_H] = ptvs->y[ptvs->index_x_H];
    ptv->dy[ptv->index_x_H] = ptvs->dy[ptvs->index_x_H];
  }
  if(evolves_xHe){
    ptv->y[ptv->index_x_He] = ptvs->y[ptvs->index_x_He];
    ptv->dy[ptv->index_x_He] = ptvs->dy[ptvs->index_x_He];
  }

  /* lower value */
  z_inf = 0.;

  /* minimum possible reionization redshift */
  ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_redshift] = z_inf;
  /* minimum possible starting redshift */
  ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] = ppr->reionization_start_factor*pth->reionization_width;
  if(ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] < pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width){
      ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] = pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width;
  }

  if(ppr->evolver == rk){
    generic_evolver = evolver_rk;
  }
  else{
    generic_evolver = evolver_ndf15;
  }

  /* Calculate a first ionization history  at upper limit */
  class_call(generic_evolver(thermodynamics_solve_derivs,
                             mz_ini,
                             mz_end,
                             ptv->y,
                             ptv->used_in_output,
                             ptv->tv_size,
                             ptpaw,
                             ppr->tol_thermo_integration,
                             ppr->smallest_allowed_variation,
                             thermodynamics_solve_timescale,  // timescale
                             1., // stepsize
                             mz_output, // values of z for output
                             Nz, // size of previous array
                             thermodynamics_solve_store_sources, // function for output
                             NULL, // print variables
                             pth->error_message),
             pth->error_message,
             pth->error_message);

  class_call(thermodynamics_reionization_get_tau(ppr,
                                                 pba,
                                                 pth,
                                                 ptw),
             pth->error_message,
             pth->error_message);

  tau_inf=ptw->reionization_optical_depth;

  class_test(tau_inf > pth->tau_reio,
             pth->error_message,
             "CLASS cannot reach the low value of tau_reio that was selected, even when setting z_reio as low as 0.\nThis means that some additional physical component is requiring some minimal tau_reio_min = %.10e.\nThis is usually caused by strong energy injections or other modifications of the x_e(z) behaviour.",tau_inf);

  /* Restore initial conditions */
  ptv->y[ptv->index_D_Tmat] = ptvs->y[ptvs->index_D_Tmat];
  ptv->dy[ptv->index_D_Tmat] = ptvs->dy[ptvs->index_D_Tmat];

  if(pth->has_idm_b == _TRUE_){
    ptv->y[ptv->index_T_idm_b] = ptvs->y[ptvs->index_T_idm_b]; //DCH
    ptv->dy[ptv->index_T_idm_b] = ptvs->dy[ptvs->index_T_idm_b]; //DCH
  }

  if(evolves_xH){
    ptv->y[ptv->index_x_H] = ptvs->y[ptvs->index_x_H];
    ptv->dy[ptv->index_x_H] = ptvs->dy[ptvs->index_x_H];
  }
  if(evolves_xH){
    ptv->y[ptv->index_x_He] = ptvs->y[ptvs->index_x_He];
    ptv->dy[ptv->index_x_He] = ptvs->dy[ptvs->index_x_He];
  }

  /** - try intermediate values by bisection */
  counter=0;
  while ((tau_sup-tau_inf) > pth->tau_reio * ppr->reionization_optical_depth_tol) {
    z_mid=0.5*(z_sup+z_inf);

    /* reionization redshift */
    ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_redshift] = z_mid;

    /* infer starting redshift for hygrogen (Note, that this is only the start of the ADDITIONAL tanh re-ionization function)*/
    ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] = ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_redshift]+ppr->reionization_start_factor*pth->reionization_width;
    /* if starting redshift for helium is larger, take that one
     *    (does not happen in realistic models) */
    if(ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] < pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width){
        ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] = pth->helium_fullreio_redshift+ppr->reionization_start_factor*pth->helium_fullreio_width;
    }

    class_test(ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_start] > ppr->reionization_z_start_max,
               pth->error_message,
               "starting redshift for reionization > reionization_z_start_max = %e",ppr->reionization_z_start_max);

    /* Compute a new ionization history */
    class_call(generic_evolver(thermodynamics_solve_derivs,
                               mz_ini,
                               mz_end,
                               ptv->y,
                               ptv->used_in_output,
                               ptv->tv_size,
                               ptpaw,
                               ppr->tol_thermo_integration,
                               ppr->smallest_allowed_variation,
                               thermodynamics_solve_timescale,  // timescale
                               1., // stepsize
                               mz_output, // values of z for output
                               Nz, // size of previous array
                               thermodynamics_solve_store_sources, // function for output
                               NULL, // print variables
                               pth->error_message),
               pth->error_message,
               pth->error_message);

    /* Restore initial conditions */
    ptv->y[ptv->index_D_Tmat] = ptvs->y[ptvs->index_D_Tmat];
    ptv->dy[ptv->index_D_Tmat] = ptvs->dy[ptvs->index_D_Tmat];

    if(pth->has_idm_b == _TRUE_){
      ptv->y[ptv->index_T_idm_b] = ptvs->y[ptvs->index_T_idm_b];//DCH
      ptv->dy[ptv->index_T_idm_b] = ptvs->dy[ptvs->index_T_idm_b];//DCH
    }

    if(evolves_xH){
      ptv->y[ptv->index_x_H] = ptvs->y[ptvs->index_x_H];
      ptv->dy[ptv->index_x_H] = ptvs->dy[ptvs->index_x_H];
    }
    if(evolves_xH){
      ptv->y[ptv->index_x_He] = ptvs->y[ptvs->index_x_He];
      ptv->dy[ptv->index_x_He] = ptvs->dy[ptvs->index_x_He];
    }

    class_call(thermodynamics_reionization_get_tau(ppr,
                                                   pba,
                                                   pth,
                                                   ptw),
               pth->error_message,
               pth->error_message);

    tau_mid=ptw->reionization_optical_depth;

    /* trial */
    if (tau_mid > pth->tau_reio) {
        z_sup=z_mid;
        tau_sup=tau_mid;
    }
    else {
        z_inf=z_mid;
        tau_inf=tau_mid;
    }

    counter++;
    class_test(counter > _MAX_IT_,
               pth->error_message,
               "while searching for reionization_optical_depth, maximum number of iterations exceeded");
  }

  /** - store the ionization redshift in the thermodynamics structure */
  pth->z_reio = ptw->ptrp->reionization_parameters[ptw->ptrp->index_reio_redshift];

  class_call(thermodynamics_vector_free(ptv),
             pth->error_message,
             pth->error_message);

  ptw->ptdw->tv = ptvs;

 return _SUCCESS_;

}


/**
 * Routine to get the optical depth of reionization
 *
 * @param ppr        Input: pointer to precision structure
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param ptw        Input: pointer to thermodynamics workspace
 * @return the error status
 */
int thermodynamics_reionization_get_tau(struct precision * ppr,
                                        struct background * pba,
                                        struct thermo * pth,
                                        struct thermo_workspace * ptw){

  /** Summary: */

  /** Define local variables */
  /* running index inside thermodynamics table */
  int i,integration_index;
  double x_e_min;

  x_e_min = 1e100;

  /**
   * We are searching now for the start of re-ionization.
   * Note, that the value reionization_parameters[index_reio_start]
   * is only the start of the added re-ionization tanh,
   * but not necessarily the total start of re-ionizatiom
   *
   * Re-ionization could be longer/shifted through energy injections.
   *
   * Please note, that actually the definition of tau_reio is not
   * clearly defined. We take it here to be the global minimum of
   * the free electron fraction. Note, that this is a choice!
   * */
  for(i=0;i<pth->tt_size-1;++i){
    if(pth->thermodynamics_table[i*pth->th_size+pth->index_th_xe]<x_e_min){
      x_e_min = pth->thermodynamics_table[i*pth->th_size+pth->index_th_xe];
      integration_index = i;
    }
  }

  class_test(integration_index == pth->tt_size,
             pth->error_message,
             "reionization start = %e > largest redshift in thermodynamics table",pth->z_table[i]);


  /** - --> spline \f$ d \tau / dz \f$ with respect to z in view of integrating for optical depth between 0 and the just found starting index */
  class_call(array_spline_table_line_to_line(pth->tau_table,
                                             integration_index,
                                             pth->thermodynamics_table,
                                             pth->th_size,
                                             pth->index_th_dkappa,
                                             pth->index_th_dddkappa,
                                             _SPLINE_EST_DERIV_,
                                             pth->error_message),
             pth->error_message,
             pth->error_message);

  /** - --> integrate for optical depth */
  class_call(array_integrate_all_spline_table_line_to_line(pth->tau_table,
                                                           integration_index,
                                                           pth->thermodynamics_table,
                                                           pth->th_size,
                                                           pth->index_th_dkappa,
                                                           pth->index_th_dddkappa,
                                                           &(ptw->reionization_optical_depth),
                                                           pth->error_message),
             pth->error_message,
             pth->error_message);

  ptw->reionization_optical_depth *= -1; // The tau sampling is inverted, so we have to correct for that here

  return _SUCCESS_;

}


/**
 * This function is passed to the generic evolver and is called whenever we want to store values for a given z that is passed in mz_output.
 * Depending on the current approximation scheme the ionization fraction is either computed analytically, semi-analytically or from the
 * (interpolated) output values of y. Moreover there is an automatic smoothing enabled which smoothes out the the ionization_fraction
 * after each approximation switch.
 *
 * This is one of the few functions in the code which is passed to the generic_evolver routine. Since generic_evolver
 * should work with functions passed from various modules, the format of the arguments is a bit special:
 *
 * - fixed parameters and workspaces are passed through a generic pointer. generic_evolver doesn't know the content of this
 *   pointer.
 *
 * - the error management is a bit special: errors are not written as usual to pth->error_message, but to a generic error_message passed
 *   in the list of arguments.
 *
 * All quantities are computed by a simple call to thermodynamics_solve_derivs, which computes all necessary quantities
 * and stores them in the ptdw thermo_diffeq_workspace structure
 *
 * @param mz                       Input: negative redshift
 * @param y                        Input: vector of thermodynamical quantities
 * @param dy                       Input: vector of redshift derivatives of theses quantities
 * @param index_z                  Input: index in the array mz_output
 * @param parameters_and_workspace Input/Output: in input, all parameters needed by thermodynamics_solve_derivs; in output, recombination table
 * @param error_message            Output: error message
 * @return the error status
 */
int thermodynamics_solve_store_sources(double mz,
                                       double * y,
                                       double * dy,
                                       int index_z,
                                       void * thermo_parameters_and_workspace,
                                       ErrorMsg error_message) {

  /** Summary: */

  /** Define local variables */
  /* Redshift and ionization fraction/temperature */
  double z;
  double x,Tmat;
  /* Recfast smoothing */
  double x_previous, weight,s;
  /* Structures as shorthand_notation */
  struct thermodynamics_parameters_and_workspace * ptpaw;
  struct precision * ppr;
  struct background * pba;
  struct thermo * pth;
  struct thermo_workspace * ptw;
  struct thermo_diffeq_workspace * ptdw;
  struct thermo_vector * ptv;
  int ap_current;
  struct thermorecfast * precfast;
  struct thermohyrec * phyrec;

  /* Redshift */
  z = -mz;

  /** - rename structure fields (just to avoid heavy notations) */
  /* Structs */
  ptpaw = thermo_parameters_and_workspace;
  ppr = ptpaw->ppr;
  pba = ptpaw->pba;
  pth = ptpaw->pth;
  /* Thermo workspace & vector */
  ptw = ptpaw->ptw;
  ptdw = ptw->ptdw;
  ptv = ptdw->tv;
  /* Recfast/HyRec */
  precfast = ptdw->precfast;
  phyrec = ptdw->phyrec;
  /* Approximation flag */
  ap_current = ptdw->ap_current;

  /* Tell heating it should store the heating at this z in its internal table */
  (pth->he).to_store = _TRUE_;
  if(pth->recombination == hyrec){
    phyrec->to_store = _TRUE_;
  }

  /* Recalculate all quantities at this current redshift (they are all stored in ptdw) */
  class_call(thermodynamics_solve_derivs(mz,y,dy,thermo_parameters_and_workspace,error_message),
             error_message,
             error_message);
  Tmat = y[ptv->index_D_Tmat] + ptw->Tcmb*(1.+z);

  /* Make super sure that our x is correct and uses the current derivative (see current_quantities for further comments) */
  class_call(thermodynamics_solve_current_quantities(z,y,pth,ptw,ap_current),
             pth->error_message,
             pth->error_message);
  x = ptdw->x_reio;

  /** - In the recfast case, we manually smooth the results a bit */
  if(pth->recombination == recfast){
    /* Smoothing if we are shortly after an approximation switch, i.e. if z is within 2 delta after the switch*/
    if(ap_current != 0 && z > ptdw->ap_z_limits[ap_current-1]-2*ptdw->ap_z_limits_delta[ap_current]){

      class_call(thermodynamics_solve_current_quantities(z,y,pth,ptw,ap_current-1),
                 pth->error_message,
                 pth->error_message);

      x_previous = ptdw->x_reio;
      // get s from 0 to 1
      s = (ptdw->ap_z_limits[ap_current-1]-z)/(2*ptdw->ap_z_limits_delta[ap_current]);
      // infer f2(x) = smooth function interpolating from 0 to 1
      weight = f2(s);

      x = weight*x+(1.-weight)*x_previous;
    }

  }

  /** - Store the results in the table */
  /* results are obtained in order of decreasing z, and stored in order of growing z */

  /* ionization fraction */
  pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_xe] = x;

  /* Tb */
  pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_Tb] = Tmat;

  /* Baryon temperature derivative (needed in perturbations for idm_b) DCH */
  pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_dTb] = dy[ptv->index_D_Tmat]+ptw->Tcmb;

  /* cb2 = (k_B/mu) Tb (1-1/3 dlnTb/dlna) = (k_B/mu) Tb (1 - 1/3 (1+z) dlnTb/d(-z)) */
  pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_cb2]
    = _k_B_ / ( _c_ * _c_ * _m_H_ ) * (1. + (1./_not4_ - 1.) * ptw->YHe + x * (1.-ptw->YHe)) * Tmat * (1. - (1.+z) * (dy[ptv->index_D_Tmat]+ptw->Tcmb) / Tmat / 3.);

  /* dkappa/dtau = a n_e x_e sigma_T = a^{-2} n_e(today) x_e sigma_T (in units of 1/Mpc) */
  pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_dkappa]
    = (1.+z) * (1.+z) * ptw->SIunit_nH0 * x * _sigma_ * _Mpc_over_m_;

  if(pth->has_idm_b == _TRUE_){

    /* Temperature of interacting dark matter (with baryons) T_idm_b DCH */
    pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_T_idm_b] = ptdw->T_idm_b;

    /* Derivative of temperature of interacting dark matter (with baryons) T_idm_b DCH */
    pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_dT_idm_b] = dy[ptv->index_T_idm_b];

    /* c_idm_b^2 is the dark matter sound speed for idm_b DCH */
    pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_c_idm_b2] = ptdw->c_idm_b2;

    /* Dark Matter - Baryon interaction coefficient DCH */
    pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_R_idm_b] = ptdw->R_idm_b;

    /* Derivative (with respect to z) of Dark Matter - Baryon interaction coefficient DCH */
    pth->thermodynamics_table[(pth->tt_size-index_z-1)*pth->th_size+pth->index_th_R_idm_b_prime] = ptdw->R_idm_b_prime;

  }

  return _SUCCESS_;

}

/**
 * This function is just for book-keeping of the evolvers. The rkck needs an actual timescale of the evolved quantities,
 * but the ndf15 does not.
 *
 * @param z             Input: redshift
 * @param thermo_...    Input: pointer to parameters and workspace
 * @param timescale     Output: pointer to the timescale
 * @param error_message Output: possible errors are written here
 * @return the error status
 */
int thermodynamics_solve_timescale(double z,
                                   void * thermo_parameters_and_workspace,
                                   double * timescale,
                                   ErrorMsg error_message){
  *timescale = 1.;
  return _SUCCESS_;

}


/**
 * Function for formatting the titles to be output
 *
 * @param pba        Input: pointer to background structure
 * @param pth        Input: pointer to the thermodynamics structure
 * @param titles     Input: titles string containing all titles
 * @return the error status
 */
int thermodynamics_output_titles(struct background * pba,
                                 struct thermo *pth,
                                 char titles[_MAXTITLESTRINGLENGTH_]) {

  class_store_columntitle(titles,"z",_TRUE_);
  class_store_columntitle(titles,"conf. time [Mpc]",_TRUE_);
  class_store_columntitle(titles,"x_e",_TRUE_);
  class_store_columntitle(titles,"kappa' [Mpc^-1]",_TRUE_);
  //class_store_columntitle(titles,"kappa''",_TRUE_);
  //class_store_columntitle(titles,"kappa'''",_TRUE_);
  class_store_columntitle(titles,"exp(-kappa)",_TRUE_);
  class_store_columntitle(titles,"g [Mpc^-1]",_TRUE_);
  //class_store_columntitle(titles,"g'",_TRUE_);
  //class_store_columntitle(titles,"g''",_TRUE_);
  class_store_columntitle(titles,"Tb [K]",_TRUE_);
  class_store_columntitle(titles,"dTb [K]",_TRUE_);
  class_store_columntitle(titles,"c_b^2",_TRUE_);
  if(pth->has_idm_b==_TRUE_){ //DCH
    class_store_columntitle(titles,"Tidm_b [K]",_TRUE_);
    class_store_columntitle(titles,"dTidm_b [K]",_TRUE_);
    class_store_columntitle(titles,"c_idm_b^2",_TRUE_);
    class_store_columntitle(titles,"R_idm_b ",_TRUE_);
  }
  class_store_columntitle(titles,"tau_d",_TRUE_);
  //class_store_columntitle(titles,"max. rate",_TRUE_);
  class_store_columntitle(titles,"r_d",pth->compute_damping_scale);

  return _SUCCESS_;

}


/**
 * Output the data for the output into files
 *
 * @param pba                 Input: pointer to background structure
 * @param pth                 Input: pointer to the thermodynamics structure
 * @param number_of_titles    Input: number of titles
 * @param data                Input: pointer to data file
 * @return the error status
 */
int thermodynamics_output_data(struct background * pba,
                               struct thermo *pth,
                               int number_of_titles,
                               double *data){

  int index_z, storeidx;
  double *dataptr, *pvecthermo;
  double z,tau;

  // pth->number_of_thermodynamics_titles = get_number_of_titles(pth->thermodynamics_titles);
  // pth->size_thermodynamics_data = pth->number_of_thermodynamics_titles*pth->tt_size;

  /* Store quantities: */
  for (index_z=0; index_z<pth->tt_size; index_z++){
    dataptr = data + index_z*number_of_titles;
    pvecthermo = pth->thermodynamics_table+index_z*pth->th_size;
    z = pth->z_table[index_z];
    storeidx=0;

    class_call(background_tau_of_z(
                                   pba,
                                   z,
                                   &tau
                                   ),
               pba->error_message,
               pth->error_message);

    class_store_double(dataptr,z,_TRUE_,storeidx);
    class_store_double(dataptr,tau,_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_xe],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_dkappa],_TRUE_,storeidx);
    //class_store_double(dataptr,pvecthermo[pth->index_th_ddkappa],_TRUE_,storeidx);
    //class_store_double(dataptr,pvecthermo[pth->index_th_dddkappa],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_exp_m_kappa],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_g],_TRUE_,storeidx);
    //class_store_double(dataptr,pvecthermo[pth->index_th_dg],_TRUE_,storeidx);
    //class_store_double(dataptr,pvecthermo[pth->index_th_ddg],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_Tb],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_dTb],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_cb2],_TRUE_,storeidx);
    if(pth->has_idm_b == _TRUE_){
      class_store_double(dataptr,pvecthermo[pth->index_th_T_idm_b],_TRUE_,storeidx);
      class_store_double(dataptr,pvecthermo[pth->index_th_dT_idm_b],_TRUE_,storeidx);
      class_store_double(dataptr,pvecthermo[pth->index_th_c_idm_b2],_TRUE_,storeidx);
      class_store_double(dataptr,pvecthermo[pth->index_th_R_idm_b],_TRUE_,storeidx);
    }
    class_store_double(dataptr,pvecthermo[pth->index_th_tau_d],_TRUE_,storeidx);
    //class_store_double(dataptr,pvecthermo[pth->index_th_rate],_TRUE_,storeidx);
    class_store_double(dataptr,pvecthermo[pth->index_th_r_d],pth->compute_damping_scale,storeidx);

  }

  return _SUCCESS_;

}


/* DCH document this */
/* !!! this needs adjusting to reflect the new format, especially in terms of the precision parameters*/

int input_obtain_idm_b_z_ini(
                            struct precision * ppr,
                            struct background *pba,
                            struct thermo *pth
                            ) {

  //ppr->recfast_Nz0 += ppr->recfast_Nz_idm_b;

  if(pth->n_index_idm_b < -3){
    //add special treatment here
    //ppr->recfast_z_initial = 1.e12;
  }

  else{
    double lhs=1.;
    double rhs=0.;
    double fHe = 1-pth->YHe;
    double m_b = _m_p_*_c_*_c_/_eV_;
    double Vrms_idm_b2;
    double T_diff_idm_b;
    double z_ini = 1100.;

    if (1.+z_ini> 1.e3)
      Vrms_idm_b2 = 1.e-8;
    else
      Vrms_idm_b2 = 1.e-8*pow((1.+z_ini)/1.e3,2);

    while(lhs > rhs){
      z_ini = z_ini*1.5;

      class_test(z_ini> 1.e11,
                 pth->error_message,
                 "Your DM and baryon temperatures are never the same, CLASS is thus unable to define an initial time for the integral. Please adjust your cross section.");

      T_diff_idm_b = pba->T_cmb*_k_B_/_eV_*(1.+z_ini)*(1./m_b+1./pth->m_idm) + (Vrms_idm_b2/3.0); //m and T are all in eV

      lhs = sqrt(pba->Omega0_ur+pba->Omega0_g)*pba->H0*(1.+z_ini)*(1+z_ini);

      rhs = pow(1.+z_ini,3) * pth->n_coeff_idm_b * pba->Omega0_idm_b * (pba->H0*pba->H0)*
        pth->cross_idm_b * fHe * 1/(m_b+pth->m_idm) * pow(T_diff_idm_b,(pth->n_index_idm_b+1.0)/2.0) *
        (3.e-4*pow(_c_,4.)/(8.*_PI_*_Mpc_over_m_*_G_*_eV_)); //conversion coefficient for the units

      //printf("z_ini = %e lhs = %e rhs = %e H0 = %e Tcmb = %e\n", z_ini, lhs, rhs, pba->H0, pba->T_cmb);

    }

    //if(z_ini*10 < ppr->recfast_z_initial){
      //ppr->recfast_Nz_idm_b = 0;
    //}
    //else{
      //ppr->recfast_z_initial = z_ini*50;
    //}

  }

  if (pth->thermodynamics_verbose > 0)
    //printf("the thermodynamycs integration for Tb and Tdm will start at z_ini = %e\n",ppr->recfast_z_initial);
    printf("AAAAAHHHHHHHH\n");
  return _SUCCESS_;

}
