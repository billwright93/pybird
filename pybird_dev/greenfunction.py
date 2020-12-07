import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy.integrate import odeint
from scipy import interpolate


class GreenFunction(object):

    def __init__(self, Omega0_m, w=None, quintessence=False, MG=False, Omega_rc=None, nDGP=False):
        #print(quintessence, MG, nDGP)
        self.Omega0_m = Omega0_m
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m)/self.Omega0_m
        self.wcdm = False
        self.quintessence = False
        self.MG = False
        self.nDGP = False
        #print('Om_rc:', Omega_rc)
        if w is not None:
            self.w = w
            if quintessence: self.quintessence = True
            else: self.wcdm = True
        if Omega_rc is not None: #should be elif since don't want both wcDM and nDGP?
            #print('Omega_rc specified, turning on nDGP if you have not done so already!')
            self.Omega_rc = Omega_rc
            if MG:
                if nDGP: print('Omega_rc specified, nDGP and MG selected -- good job!')
                else: print('Omega_rc specified and MG selected, but nDGP not selected -- uh oh!\nTurning on nDGP automatically.')
            else:
                if nDGP: print('Omega_rc specified and nDGP selected, but MG not selected -- uh oh!\nTurning on MG automatically.')
                else: print('Omega_rc specified, but neither nDGP and MG selected -- not good!\nTurning on both MG and nDGP automatically.')
            self.nDGP = True
            self.MG = True #needed?
        else:
            if MG:
                print('nDGP GF:', nDGP)
                if nDGP: print('nDGP and MG selected, but Omega_rc not specified -- not good!')
                else: print('MG selected, but nDGP not selected and Omega_rc not specified -- currently nDGP is the only MG model in PyBird!')
            else:
                if nDGP: print('nDGP selected, but MG not selected and Omega_rc not specified -- uh oh!')
                else: print('No modified gravity stuff specified or selected -- that is fine!')
        '''elif Omega_rc is None and nDGP==True:
            print('nDGP selected but Omega_rc not specified!\nTurning off MG and nDGP.')
            self.MG = False
            self.nDGP = False
        elif Omega_rc is not None and MG==True and nDGP==False:
            print('nDGP selected but Omega_rc not specified!')'''

        self.epsrel = 1e-4

        if self.nDGP:
            x_arr_nDGP = np.arange(-7, 7., 14/1000)
            self.Dp_nDGP = interpolate.interp1d(x_arr_nDGP,self.D_DD_MG_num(1)[0],kind='cubic')
            self.DDp_nDGP = interpolate.interp1d(x_arr_nDGP,self.D_DD_MG_num(1)[1],kind='cubic')
            self.Dm_nDGP = interpolate.interp1d(x_arr_nDGP,self.D_DD_minus_MG_num(1)[0],kind='cubic')
            self.DDm_nDGP = interpolate.interp1d(x_arr_nDGP,self.D_DD_minus_MG_num(1)[1],kind='cubic')

    def C(self, a):
        if self.quintessence: return 1. + (1.+self.w) * self.OmegaL_by_Omega_m * a**(-3.*self.w)
        else: return 1.

    def H(self, a):
        """Conformal Hubble"""
        if self.wcdm or self.quintessence: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
        else: return (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5

    def H3(self, a):
        return self.C(a)/self.H(a)**3

    def Omega_m(self, a):
        return self.Omega0_m / (self.H(a)**2 * a)

    def H_NC(self, a):
        """Non-Conformal Hubble, H0=1"""
        #if self.nDGP: return np.sqrt(Omega0_m/a/a/a + ((1.-Omega0_m)+ 2.*Omega_rc*(np.sqrt((Omega0_m/a/a/a)/Omega_rc + 1.) - 1.)) + Omega_rc) - np.sqrt(Omega_rc) #nDGP+DE
        #else: return (Omega0_m/a/a/a + (1.-Omega0_m))**0.5 #LCDM
        return (self.Omega0_m/a/a/a + (1.-self.Omega0_m))**0.5 #expansion fixed to LCDM

    def dHda_NC(self, a):
        """Derivative of non-Conformal Hubble w.r.t. a, H0=1"""
        #if self.nDGP: return (-3.*a**(-1. - 3.*(1 + w))*(1 + w)*(1.-Omega0_m) - 3*Omega0_m*a**(-4))/(2.*np.sqrt((1.-Omega0_m)*a**(-3*(1 + w)) + Omega0_m*a**(-3) + Omega_rc)) #nDGP+DE
        #else: return 0.5*(-3.*Omega0_m/a/a/a/a)*(Omega0_m/a/a/a + (1.-Omega0_m))**(-0.5) #LCDM
        return 0.5*(-3.*self.Omega0_m/a/a/a/a)*(self.Omega0_m/a/a/a + (1.-self.Omega0_m))**(-0.5) #expansion fixed to LCDM

    
    def H_x(self,x):
        """H using the time variable x = log(a)"""
        """Useful for numerical estimation of D and f"""
        return (self.Omega0_m*np.e**(-3.*x) + 1. - self.Omega0_m)**0.5 #background expansion fixed to LCDM
    
    def Om_x(self,x):
        """Matter evolution using the time variable x = log(a)"""
        return (self.Omega0_m)*(np.e**(-3*x))/(self.H_x(x)**2.)
        
    def dlogHdx(self,x):
        """Log Derivative of H w.r.t. the time variable x = log(a)"""
        """Useful for numerical estimation of D and f"""
        return 3./2.*(-1 + (1-self.Omega0_m)/(1 + self.Omega0_m*(-1 + np.e**(-3*x)))) #background expansion fixed to LCDM
    
    def beta_x(self,x):
        """beta function of nDGP, time variable x = log(a)"""
        return 1 + self.H_x(x)/(np.sqrt(self.Omega_rc))*(1 + self.dlogHdx(x)/3)
        
    def mu_x(self,x):
        """mu function of nDGP, time variable x = log(a)"""
        return 1 + 1/(3*self.beta_x(x))
    
    def beta(self, a):
        if self.nDGP: return 1. + self.H_NC(a)*(1.+a*self.dHda_NC(a)/3./self.H_NC(a))/np.sqrt(self.Omega_rc)
        else: return 1.

    def mu(self, a):
        if self.nDGP: return 1.+1./(3.*self.beta(a))
        else: return 1.
        

    def mu2(self, a):
        if self.nDGP: return (-0.5*self.H_NC(a)**2.*(1./(3.*self.beta(a)))**3./self.Omega_rc)
        else: return 0.

    def mu22(self, a):
        if self.nDGP: return (0.5*self.H_NC(a)**4.*(1./(3.*self.beta(a)))**5./self.Omega_rc/self.Omega_rc)
        else: return 0.

    def compute_primes_MG(self, Y, x):
        """Second order differential equation for growth factor D extended from Eq.(A.1) of 2005.04805"""
        """The equation is solved in the time variable x = log(a)"""
        D, DD = Y
        DDD = [DD,- (2. + self.dlogHdx(x))*DD + 3./2.*self.mu_x(x)*self.Om_x(x)*D]
        return DDD

#--------------------- THIS IS THE PART MARCO MODIFIED - BEGINNING ----------------------#


    def D_DD_MG_num(self,a):
        """Solve for growth and decay mode in general cosmologies"""
        """Only time-dependent growth factor"""
        """Time variable x = log(a)"""
        xin = -7.
        xfin = +7.
        xpoints = 1000
        if a < np.e**xin:
            print('Need to decrease a_ini from ', np.e**xin, ' to below ', a)
        delta_x = (xfin - xin)/xpoints
        xss = np.arange(xin,xfin,delta_x) # x's for the growth mode solution
        xss_inv = xss[::-1] # inverted a's for the decay mode solution
        
        #Initial conoditions
        D0plus = [np.e**xin,np.e**xin]
        #Numerical solutions
        ans_plus = odeint(self.compute_primes_MG,D0plus,xss,mxstep = 4000)
        Dplus = ans_plus[:,0] #EARLY time initial conditions (EdS approx. is sufficient) for the growing mode
        DDplus = ans_plus[:,1] #LATE time initial conditions (EdS approx. is sufficient) for the decay mode
        
        
        #Dp = interpolate.interp1d(xss,Dplus,kind = 'cubic')
        #DDp = interpolate.interp1d(xss,DDplus,kind = 'cubic')
        return Dplus,DDplus#Dp(np.log(a)),DDp(np.log(a))
        
    def D_DD_minus_MG_num(self,a):
        """Solve for decay mode in general cosmologies"""
        """Only time-dependent growth factor"""
        """Time variable x = log(a)"""
        xin = -7.
        xfin = +7.
        xpoints = 1000
        if a < np.e**xin:
            print('Need to decrease a_ini from ', np.e**xin, ' to below ', a)
        delta_x = (xfin - xin)/xpoints
        xss = np.arange(xin,xfin,delta_x) # x's for the growth mode solution
        xss_inv = xss[::-1] # inverted a's for the decay mode solution
        
        #Initial conditions
        D0minus = [np.e**(-2*xfin),-2*np.e**(-2*xfin)]
        #Numerical solutions
        ans_minus = odeint(self.compute_primes_MG,D0minus,xss_inv,mxstep = 4000)
        Dminus1 = ans_minus[:,0][::-1]
        DDminus = ans_minus[:,1][::-1]
        Dminus = Dminus1/(Dminus1[0]/(np.e**(-3*xin/2)))
        DDminus = DDminus/(Dminus1[0]/(np.e**(-3*xin/2)))
        #Dm = interpolate.interp1d(xss,Dminus,kind = 'cubic')
        #DDm = interpolate.interp1d(xss,DDminus,kind = 'cubic')
        return Dminus,DDminus#Dm(np.log(a)),DDm(np.log(a))


    
    
#--------------------- THIS IS THE PART MARCO MODIFIED - ENDING -------------------------#
    

#--------- OLD PART, REMOVE ------------------------#
#    def D_DD_MG_num(self, a):
#        """Solve for growth mode"""
#        a_ini = 1e-7
#        if a < a_ini:
#            print('Need to decrease a_ini from ', a_ini, ' to below ', a)
#        D_ini_growth = a_ini
#        DD_ini_growth = 1.
#        a_points = 2 #100
#        a_arr = np.linspace(a_ini, a, a_points)
#        Y_ini_growth = [D_ini_growth, DD_ini_growth]
#        ans = odeint(self.compute_primes_MG, Y_ini_growth, a_arr)
#        D_growth = ans[-1,0]
#        DD_growth = ans[-1,1]
#        #print(self.Omega_rc)
#        return D_growth, DD_growth

#    def D_DD_minus_MG_num(self, a):
#        """Solve for decay mode"""
#        a_ini = 1e-7
#        if a < a_ini:
#            print('Need to decrease a_ini from ', a_ini, ' to below ', a)
#        D_ini_decay = a_ini**(-1.5)
#        DD_ini_decay = -1.5*a_ini**(-2.5)
#        a_points = 2 #100
#        a_arr = np.linspace(a_ini, a, a_points)
#        Y_ini_decay = [D_ini_decay, DD_ini_decay]
#        ans = odeint(self.compute_primes_MG, Y_ini_decay, a_arr)
#        D_decay = ans[-1,0]
#        DD_decay = ans[-1,1]
#        return D_decay, DD_decay

#--------- OLD PART, REMOVE ------------------------#
    
    
    def D(self, a):
        """Growth factor"""
        if self.MG: return self.Dp_nDGP(np.log(a))#/self.Dp_nDGP(np.log(1))
        elif self.wcdm: return a*hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-(a**(-3*self.w))*self.OmegaL_by_Omega_m)#/(hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-self.OmegaL_by_Omega_m))
        else:
            I = quad(self.H3, 0, a, epsrel=self.epsrel)[0]
            return 5 * self.Omega0_m * I * self.H(a) / (2.*a)

    def DD(self, a):
        """Derivative of growth factor"""
        if self.MG: return self.DDp_nDGP(np.log(a))/a#/self.Dp_nDGP(np.log(1))
        elif self.wcdm: return (-(a**(-3.*self.w))*self.OmegaL_by_Omega_m*((3*(self.w-1))/(6.*self.w-5.))*hyp2f1(1.5-0.5*(1/self.w),1-(1/(3.*self.w)),2-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)+hyp2f1((self.w-1)/(2.*self.w),-1/(3.*self.w),1-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m))#/(hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-self.OmegaL_by_Omega_m))
        else: return (2.5-(1.5*self.D(a)/a)) * self.Omega_m(a) * self.C(a)

    def fplus(self, a):
        """Growth rate"""
        return a * self.DD(a) / self.D(a)

    def Dminus(self, a):
        """Decay factor"""
        if self.MG: return self.Dm_nDGP(np.log(a))#/self.Dm_nDGP(np.log(1))
        elif self.wcdm: return a**(-3/2.)*hyp2f1(1/(2.*self.w),(1/2.)+(1/(3.*self.w)),1+(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        else: return self.H(a) / (a*self.Omega0_m**.5)

    def DDminus(self, a):
        """Derivative of decay factor"""
        if self.MG: return self.DDm_nDGP(np.log(a))/a#/self.Dm_nDGP(np.log(1))
        elif self.wcdm: return ((-1+3.*self.w)*hyp2f1(0.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m))-(2+3.*self.w)*hyp2f1(1.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m)))/(2*(a**(5/2.)))
        else: return -1.5 * self.Omega_m(a) * self.Dminus(a) / a * self.C(a)

    def fminus(self, a):
        """Decay rate"""
        return a * self.DDminus(a) / self.Dminus(a)

    def W(self, a):
        """Wronskian"""
        return self.DDminus(a) * self.D(a) - self.DD(a) * self.Dminus(a)

    #greens functions
    def G1d(self, a, ai):
        return(self.DDminus(ai)*self.D(a)-self.DD(ai)*self.Dminus(a))/(ai*self.W(ai))
    def G2d(self, a, ai):
        return self.fplus(ai)*(self.Dminus(a)*self.D(ai)-self.D(a)*self.Dminus(ai))/(ai*ai*self.W(ai))
    def G1t(self, a, ai):
        return a*(self.DDminus(ai)*self.DD(a)-self.DD(ai)*self.DDminus(a))/(self.fplus(a)*ai*self.W(ai))
    def G2t(self, a, ai):
        return a*self.fplus(ai)*(self.DDminus(a)*self.D(ai)-self.DD(a)*self.Dminus(ai))/(self.fplus(a)*ai*ai*self.W(ai))

    # second order coefficients
    def I1d(self, ai, a):
        #print(ai, a, (self.G1d(a,ai)*self.fplus(ai) + self.G2d(a,ai)*self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a))
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai) + self.G2d(a,ai)*self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(a)
    def I2d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai) - self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(a)
    def I1t(self, ai, a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai) + self.G2t(a,ai)*self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(a)
    def I2t(self, ai, a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai) - self.mu2(ai)*(1.5*self.Omega_m(ai))**2/self.fplus(ai))*self.D(ai)**2/self.D(a)**2 / self.C(a)
        else: return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2 / self.C(a)

    # second order time integrals
    def mG1d(self, a):
        return quad(self.I1d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0] # set the lower limit to a very small but not null value
    def mG2d(self, a):
        return quad(self.I2d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        return quad(self.I1t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        return quad(self.I2t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence/MG time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG1d(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2d(self, ai, a):
        if self.MG: return ( self.G1d(a,ai)*self.fplus(ai)*self.mG2d(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai) )/self.fplus(ai) )*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU1t(self, ai, a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG1d(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IU2t(self, ai, a):
        if self.MG: return ( self.G1t(a,ai)*self.fplus(ai)*self.mG2d(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*( self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai) )/self.fplus(ai) )*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG1t(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai)*self.mG1t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21d(self, ai, a):
        if self.MG: return (self.G1d(a,ai)*self.fplus(ai)*self.mG2t(ai) + self.G2d(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22d(self, ai, a):
        if self.MG: return self.G2d(a,ai)*(self.fplus(ai)*self.mG2t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    def IV11t(self, ai,a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG1t(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV12t(self, ai,a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai)*self.mG1t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG1d(ai) + 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV21t(self, ai,a):
        if self.MG: return (self.G1t(a,ai)*self.fplus(ai)*self.mG2t(ai) + self.G2t(a,ai)*(1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)
    def IV22t(self, ai,a):
        if self.MG: return self.G2t(a,ai)*(self.fplus(ai)*self.mG2t(ai) - (1.5*self.Omega_m(ai))**2*(self.mu2(ai)*self.mG2d(ai) - 0.5*self.mu22(ai)*1.5*self.Omega_m(ai))/self.fplus(ai))*(self.D(ai)/self.D(a))**3 / self.C(a)
        else: return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(a)

    # third order time integrals
    def mU1d(self, a):
        return quad(self.IU1d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        return quad(self.IU2d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        return quad(self.IU1t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        return quad(self.IU2t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        return quad(self.IV11d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        return quad(self.IV12d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        return quad(self.IV21d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        return quad(self.IV22d,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        return quad(self.IV11t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        return quad(self.IV12t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        return quad(self.IV21t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        return quad(self.IV22t,np.e**(-7.),a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        return -3/14. + self.mV11d(a) + self.mV12d(a)
