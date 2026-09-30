import numpy as np

# RSD parameters
#bbpars = [1.,1.,1., 1., 1.]
#betapars = [1.2,1.2,1.2,1.2, 1.2]
#bvpars = [1.,1.,1.,1., 1.]

# Bias parameters
#zzarrbias = [0., 1.1, 1.4, 1.7, 2., 2.5, 3., 4.]
zzarrbias = [1.6, 2., 2.5, 3., 4.]

#meandensarr = [0.6, 0.6, 0.6, 0.6, 0.5, 0.4] 
#meandensarr = [0.8, 0.7, 0.5, 0.3, 0.25, 0.2]
meandensarr = [1.,1.,1.,1.,1.] 
renormarr = [5.5, 5.5, 2.5, 1., 0.1]

def make_pars_list():

    # Knots
    wnmean=[]; walpha=[]; wbeta=[]; wdth=[]; wrhoeps=[]; weps=[]
    
    # ******************************************************
    # ********* z=1.1 **************************************

    nmean=7.870739574; alpha = 0.51550673;   beta = 48.88968725;  dth = 0.13947851; rho_eps = 1.02857884; eps = 1.61955375                                                

    # Do it one first time for the lowest z interpolation node
    #wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)
    #wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)

    # ******************************************************
    # ********* z=1.4 **************************************

    nmean=2.4631715; alpha = 0.7156661;   beta = 4.7666552;  dth = 0.1248781; rho_eps = 1.74719952; eps = 2.58337194                                                                  

    # Do it one first time for the lowest z interpolation node
    #wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)

    # ******************************************************
    # ********* z=1.7 **************************************

    nmean=0.314436310; alpha = 2.37169129;   beta = 0.86496957;  dth = -0.50693983; rho_eps = 8.16711823; eps = 1.50266246                                                                  

    # Do it one first time for the lowest z interpolation node
    #wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)

    # ******************************************************
    # ********* z=2.0 **************************************

    #nmean=0.1608119; alpha = 2.55470012;   beta = 1.49311803;  dth = -0.91600249; rho_eps = 14.12213702; eps = 2.02903172
    nmean=3.203089; alpha = 1.97320941;   beta = 3.8450873;  dth = 0.20919576; rho_eps = 0.89166006; eps = 2.41995644 

    # Do it one first time for the lowest z interpolation node
    wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)
    wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)

    # ******************************************************
    # ********* z=2.5 **************************************

    nmean=3.203089; alpha = 1.97320941;   beta = 3.8450873;  dth = 0.20919576; rho_eps = 0.89166006; eps = 2.41995644                                                                 

    # Do it one first time for the lowest z interpolation node
    wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)

     # ******************************************************
    # ********* z=3.0 **************************************

    #nmean=3454.8186; alpha = 2.64685629;   beta = 58.66502166;  dth = 0.26010453; rho_eps = 0.33507574; eps = 1.63869552                                                                 

    # Do it one last time for the highest z interpolation node
    wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)
    wnmean.append(nmean); walpha.append(alpha); wbeta.append(beta); wdth.append(dth); wrhoeps.append(rho_eps); weps.append(eps)
    
    nmean_arr = np.array(wnmean)
    alpha_arr = np.array(walpha)
    beta_arr = np.array(wbeta)
    dth_arr = np.array(wdth)
    rhoeps_arr = np.array(wrhoeps)
    eps_arr = np.array(weps)


    # ***********************************
    # RSD parameters
    bv_arr = np.array([1.2*0.87674972, 1.2*0.87674972, 0.9*0.87674972, 0.75*0.87674972, 0.75*0.87674972])
    bb_arr = np.array([0.50433262,0.50433262,0.50433262,0.50433262,0.5043326])
    betarsd_arr = np.array([0.95137647,0.95137647,0.95137647,0.95137647,0.95137647])
    gamma_arr = np.array([1.76038262,1.76038262,1.76038262,1.76038262,1.76038262])

    return nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, bv_arr, bb_arr, betarsd_arr, gamma_arr
    #return wnmean11,walpha11,wbeta11,wdth11,wrhoeps11,weps11,wnmean12,walpha12,wbeta12,wdth12,wrhoeps12,weps12,wnmean13,walpha13,wbeta13,wdth13,wrhoeps13,weps13,wnmean14,walpha14,wbeta14,wdth14,wrhoeps14,weps14,wnmean21,walpha21,wbeta21,wdth21,wrhoeps21,weps21,wnmean22,walpha22,wbeta22,wdth22,wrhoeps22,weps22,wnmean23,walpha23,wbeta23,wdth23,wrhoeps23,weps23,wnmean24,walpha24,wbeta24,wdth24,wrhoeps24,weps24,wnmean31,walpha31,wbeta31,wdth31,wrhoeps31,weps31,wnmean32,walpha32,wbeta32,wdth32,wrhoeps32,weps32,wnmean33,walpha33,wbeta33,wdth33,wrhoeps33,weps33,wnmean34,walpha34,wbeta34,wdth34,wrhoeps34,weps34,wnmean41,walpha41,wbeta41,wdth41,wrhoeps41,weps41,wnmean42,walpha42,wbeta42,wdth42,wrhoeps42,weps42,wnmean43,walpha43,wbeta43,wdth43,wrhoeps43,weps43,wnmean44,walpha44,wbeta44,wdth44,wrhoeps44,weps44

#wnmean11,walpha11,wbeta11,wdth11,wrhoeps11,weps11,wnmean12,walpha12,wbeta12,wdth12,wrhoeps12,weps12,wnmean13,walpha13,wbeta13,wdth13,wrhoeps13,weps13,wnmean14,walpha14,wbeta14,wdth14,wrhoeps14,weps14,wnmean21,walpha21,wbeta21,wdth21,wrhoeps21,weps21,wnmean22,walpha22,wbeta22,wdth22,wrhoeps22,weps22,wnmean23,walpha23,wbeta23,wdth23,wrhoeps23,weps23,wnmean24,walpha24,wbeta24,wdth24,wrhoeps24,weps24,wnmean31,walpha31,wbeta31,wdth31,wrhoeps31,weps31,wnmean32,walpha32,wbeta32,wdth32,wrhoeps32,weps32,wnmean33,walpha33,wbeta33,wdth33,wrhoeps33,weps33,wnmean34,walpha34,wbeta34,wdth34,wrhoeps34,weps34,wnmean41,walpha41,wbeta41,wdth41,wrhoeps41,weps41,wnmean42,walpha42,wbeta42,wdth42,wrhoeps42,weps42,wnmean43,walpha43,wbeta43,wdth43,wrhoeps43,weps43,wnmean44,walpha44,wbeta44,wdth44,wrhoeps44,weps44 = make_pars_list() 
nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr, bv_arr, bb_arr, betarsd_arr, gamma_arr = make_pars_list() 
