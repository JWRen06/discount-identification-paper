"""Black-box validation provider: solves a 2-state discounted Bellman equation.

No impulse-kernel or inverse-identification formulas are used in this module.
The auditor receives aggregate observations, costs and noise calibration only.
This is an independently implemented controlled validation, not market data.
"""
import numpy as np

def solve_policy(alpha,beta,cost,persistence):
    A=np.array([[0.,0.],[0.,persistence]])
    B=np.array([1.,0.])
    Q=np.diag([cost,alpha]);N=np.array([-cost,-alpha]);R=alpha+cost
    P=np.zeros((2,2))
    for iteration in range(20000):
        cross=N+beta*A.T@P@B
        denom=R+beta*B@P@B
        nextP=Q+beta*A.T@P@A-np.outer(cross,cross)/denom
        if np.max(np.abs(nextP-P))<2e-14*max(1.,np.max(np.abs(nextP))):
            P=nextP;break
        P=nextP
    else:raise ArithmeticError('Bellman iteration failed')
    cross=N+beta*A.T@P@B;denom=R+beta*B@P@B
    gain=-cross/denom
    residual=Q+beta*A.T@P@A-np.outer(cross,cross)/denom-P
    assert np.max(np.abs(residual))<1e-10*max(1.,alpha,cost)
    return gain,iteration+1,float(np.max(np.abs(residual)))

def paired_impulse(alpha,beta,costs,pool,report_filter,lags=8):
    observations=[];iterations=0;max_residual=0.
    for cost in costs:
        outputs=[]
        gains=[]
        for s,w in pool:
            gain,it,res=solve_policy(alpha,beta,cost,s)
            gains.append(gain);iterations=max(iterations,it);max_residual=max(max_residual,res)
        for sign in [-1.,1.]:
            aggregate=np.zeros(lags)
            for j,((s,w),gain) in enumerate(zip(pool,gains)):
                d=.3*(j+1)/len(pool);p=-.2*(j+1)/len(pool)
                for n in range(lags):
                    d=s*d+(1-s)*(sign if n==0 else 0.)
                    p=float(gain@np.array([p,d]));aggregate[n]+=w*p
            outputs.append(np.convolve(aggregate,report_filter)[:lags])
        observations.append((outputs[1]-outputs[0])/2)
    return np.array(observations),dict(max_iterations=iterations,max_bellman_residual=max_residual)

def task_loss(alpha,controller_beta,task_beta,cost,pool,steps=800,innovation_variance=.04):
    """Expected discounted mission loss from a unit initial risk exposure.

    State x=(previous position,current target). Future common input is iid,
    zero mean with the given variance. Evaluate each private rule's loss, with
    positive aggregate weights, under the *same* externally specified clock.
    The reporting filter is not part of this private objective.
    """
    total=0.
    for s,w in pool:
        gain,_,_=solve_policy(alpha,controller_beta,cost,s)
        T=np.array([[gain[0],gain[1]],[0.,s]])
        innovation=np.diag([0.,(1-s)**2*innovation_variance])
        tracking= gain-np.array([0.,1.]);adjustment=gain-np.array([1.,0.])
        H=.5*(alpha*np.outer(tracking,tracking)+cost*np.outer(adjustment,adjustment))
        covariance=np.array([[0.,0.],[0.,1.]])
        local=0.
        for t in range(steps):
            local+=task_beta**t*np.sum(H*covariance)
            covariance=T@covariance@T.T+innovation
        total+=w*local
    return float(total)
