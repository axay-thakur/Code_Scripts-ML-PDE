def rk4singlestep(rhs,dt,t,u,*args):
  k1 = rhs(t,u,*args)
  k2 = rhs(t+dt/2,u+k1*dt/2,*args)
  k3 = rhs(t+dt/2,u+k2*dt/2,*args)
  k4 = rhs(t+dt,u+k3*dt,*args)
  unew = u + (dt/6)*(k1+2*k2+2*k3+k4) 
  return unew

#forward euler
def feuler(rhs,dt,t,u,*args):
    return u + dt*rhs(t,u,*args)