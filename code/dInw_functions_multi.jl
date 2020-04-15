using DelimitedFiles
using Random
using Statistics

#cd("C:/Users/Marius/Documents/INW/DecisionInfluenceNetwork/code")
include("./dInw_structures_multi.jl")


function train!(
        dInw::DecisionInfluenceNetwork;
        p::_param = _param(),
        g_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
        g_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
        g_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
        mom_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
        mom_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
        mom_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0))::Nothing
    trainer::_trainer = _trainer(dInw.inw, g_init = g_init, g_in = g_in, g_out = g_out, mom_init = mom_init, mom_in = mom_in, mom_out = mom_out)
    internal::_internal = _internal(dInw.inw)
    predictor::_predictor = _predictor(dInw.inw, dInw.data)
    n::Int = dInw.data.n
    mse::Float32 = 1f0
    mse_::Float32 = 0f0
    batchIds::Array{Int, 1} = p.batchSize == -1 ? 
        [1, n-p.testSize+1] : vcat(collect(1:p.batchSize:(n-p.testSize-1)), [n-p.testSize+1])
    batches::Array{Int, 1} = shuffle(p.testSize+1:n)
    for i in 1:p.maxIter
        for j in 1:length(batchIds)-1
            _train!(
                dInw.inw, 
                dInw.data.X[batches[batchIds[j]:batchIds[j+1]-1],:], 
                dInw.data.Y[batches[batchIds[j]:batchIds[j+1]-1],:], 
                trainer, 
                internal, 
                p)
        end
        batches .= shuffle(batches)
        if i%p.feedback == 0 
            predict!(
                predictor,
                dInw.data,
                dInw.inw)
            if p.testSize != 0
                mse_ = sum((dInw.data.Y[1:p.testSize,:] .- predictor.out[1:p.testSize,:]) .^ 2, dims = (1,2))[1]/(p.testSize*dInw.inw.m)
                print(string(
                    "Finished Iteration ", string(i), 
                    "\nMSE Training Data: ", string(sum((dInw.data.Y[p.testSize+1:end,:] .- predictor.out[p.testSize+1:end,:]) .^ 2, dims = (1,2))[1]/((n-p.testSize)*dInw.inw.m)), 
                    "\nMSE Test Data:     ", string(mse_),
                    "\n\n"))
                if ((mse - mse_) > p.thresh) | (i <= p.minIter)
                    mse = mse_
                else 
                    break 
                end
            else
                print(string(
                    "Finished Iteration ", string(i), "\nMSE: ", sum((dInw.data.Y .- predictor.out) .^ 2, dims = (1,2))[1]/n,  "\n\n"))
            end
        end
    end
    return nothing
end


function cross_train!(
        dInw::DecisionInfluenceNetwork,
        k::Int;
        z_crit::Float32 = -1f0,
        p::_param = _param())::Nothing
    n::Int = dInw.data.n
    m::Int = dInw.inw.n_
    subset_::Array{Bool, 1} = Array{Bool, 1}(undef, n)
    trainer::_trainer = _trainer(dInw.inw)
    internal::_internal = _internal(dInw.inw)
    predictor::_predictor = _predictor(dInw.inw, dInw.data)
    w_init_::Array{Float32, 2} = Array{Float32, 2}(undef, m, k) 
    w_in_::Array{Float32, 3} = Array{Float32, 3}(undef, m, m, k) 
    w_out_::Array{Float32, 2} = Array{Float32, 2}(undef, m, k) 
    sub_ids::Array{Int, 1} = Int.(floor.([i for i in 0:(n/k):n])).+1
    for i in 1:k
        print(string(
            "Starting Round ", string(i), "\n\n"))
        subset_ .= [.!(j in sub_ids[i]:sub_ids[i+1]-1) for j in 1:n]
        n_sub::Int = sum(subset_)
        mse::Float32 = 1.0f0
        mse_::Float32 = 1.0f0
        for j in 1:p.maxIter
            _train!(
                dInw.inw,
                dInw.data.X[subset_,:],
                dInw.data.Y[subset_],
                trainer,
                internal,
                p)
            if j%p.feedback == 0 
                predict!(
                    predictor,
                    dInw.data,
                    dInw.inw)
                mse_ = sum((dInw.data.Y[.!subset_] .- predictor.out[.!subset_]) .^ 2)/(n-n_sub)
                print(string(
                    "Finished Iteration ", string(j), 
                    "\nMSE Training Data: ", string(sum((dInw.data.Y[subset_] .- predictor.out[subset_]) .^ 2)/n_sub), 
                    "\nMSE Test Data:     ", string(mse_),
                    "\n\n"))
                if ((mse - mse_) > p.thresh) | (j <= p.minIter)
                    w_init_[:,i] .= dInw.inw.w_init
                    w_in_[:,:,i] .= dInw.inw.w_in
                    w_out_[:,i] .= dInw.inw.w_out
                    mse = mse_
                else
                    break
                end
            end
        end
        dInw.inw.w_init .= (rand(Float32, m) .- 0.5f0) .* dInw.inw.scale 
        dInw.inw.w_in .= (rand(Float32, m, m) .- 0.5f0) .* dInw.inw.free .* dInw.inw.scale
        dInw.inw.w_out .= (rand(Float32, m) .- 0.5f0) .* dInw.inw.scale 
        trainer.mom_init .= 0.0f0
        trainer.mom_in .= 0.0f0
        trainer.mom_out .= 0.0f0
    end
    dInw.inw.w_init .= mean(w_init_, dims = 2)[:]
    dInw.inw.w_in .= mean(w_in_, dims = 3)[:,:]
    dInw.inw.w_out .= mean(w_out_, dims = 2)[:]
    if z_crit != -1f0
        dInw.inw.free .= Float32.(abs.(dInw.inw.w_in) .>= std(w_in_, mean = dInw.inw.w_in, dims = 3)[:,:] .* z_crit)
        dInw.inw.w_in .*= dInw.inw.free
    end
    return nothing
end


function _train!(
        inw::InfluenceNetwork,
        X::Array{Int, 2},
        Y::Array{Float32, 2},
        trainer::_trainer,
        internal::_internal,
        p::_param)::Nothing
    f_::Float32 = p.batchSize .== -1 ? 1 : size(X)[1]/p.batchSize
    trainer.g_init .= 0f0
    trainer.g_in .= 0f0
    trainer.g_out .= 0f0
    trainer.upd_count_in .= 1f0
    trainer.upd_count_out .= 1f0
    for i in 1:size(Y)[1]
        # compute local gradients
        trainer.select .= X[i,:] .+ inw.index
        for j in 1:size(Y)[2]
            set_internal!(internal, inw, trainer.select, j)
            forward_prop!(internal, inw)
            internal.g_init[:,j] .= internal.g_init_'*internal.w_out
            internal.g_in[:,j] .= internal.g_in_'*internal.w_out 
            internal.g_out[:,j] .= internal.s
            internal.out[j,:] .= (internal.w_out' * internal.s) * inw.beta
        end
        back_prop!(internal, Y[i,:], inw)
        # update global gradients
        trainer.g_init[trainer.select, :] .+= internal.g_init
        trainer.g_in[trainer.select, trainer.select, :] .+= reshape(internal.g_in, inw.n, inw.n, inw.m)
        trainer.g_in .*= inw.free
        trainer.g_out[trainer.select, :] .+= internal.g_out  
        trainer.upd_count_in[trainer.select, trainer.select] .+= 1f0
        trainer.upd_count_out[trainer.select] .+= 1f0
    end
    #trainer.update_in .= trainer.upd_count_in .> 1f0
    #trainer.update_out .= trainer.upd_count_out .> 1f0
    trainer.g_init ./= trainer.upd_count_out 
    trainer.g_in ./= trainer.upd_count_in
    trainer.g_out ./= trainer.upd_count_out
    trainer.mom_init .= (trainer.mom_init .+ trainer.g_init) .* p.mom # [trainer.update_out, :]
    trainer.mom_in .= (trainer.mom_in .+ trainer.g_in) .* p.mom # [trainer.update_in, :]
    trainer.mom_out .= (trainer.mom_out .+ trainer.g_out) .* p.mom # [trainer.update_out, :]
    inw.w_init .+= f_ * p.lr * p.lr_fact .* (trainer.g_init .+ trainer.mom_init)
    inw.w_in .+= f_ * p.lr * p.lr_fact .* (trainer.g_in .+ trainer.mom_in)
    inw.w_out .+= f_ * p.lr .* (trainer.g_out .+ trainer.mom_out)
    return nothing
end


function set_internal!(
        internal::_internal,
        inw::InfluenceNetwork, 
        select::Array{Int, 1},
        j::Int)::Nothing
    internal.s .= 1f0
    internal.w_init .= inw.w_init[select, j]
    internal.w_in .= inw.w_in[select, select, j]
    internal.w_out .= inw.w_out[select, j]
    internal.g_init_ .= 0f0
    internal.g_in_ .= 0f0
    return nothing
end


function forward_prop!(
        internal::_internal,
        inw::InfluenceNetwork)::Nothing
    n = inw.n
    internal.z .= internal.w_init .* internal.s
    internal.expZ .= _expZ.(internal.z)
    internal.s .= _logit.(internal.z, internal.expZ)
    internal.g_init_[1:(n+1):end] .= 2f0 .* internal.s .* (1f0.-internal.s)
    internal.s .= internal.s .* 2f0 .- 1f0
    for t in 1:inw.rounds
        internal.s[n] = inw.bias_in
        internal.z .= internal.w_in' * internal.s                        # calculate update scores
        internal.expZ .= _expZ.(internal.z)
        internal.dnom .= _dnom.(internal.s, internal.z, internal.expZ, inw.c)      # calculate denominater function values
        internal.num .= _num.(internal.s, internal.z, internal.expZ, inw.c)
        # Update w_init gradient
        internal.t_1 .= init_t_1_.(internal.z, internal.expZ, internal.num, internal.dnom, inw.c)
        internal.t_2 .= init_t_2_.(internal.s, internal.z, internal.expZ, internal.num, internal.dnom, inw.c)
        internal.w_in[1:(n+1):end] .= internal.t_1 ./ internal.t_2
        internal.g_init_ .= internal.w_in' * internal.g_init_
        internal.g_init_ .*= internal.t_2 ./ internal.dnom.^2
        # Update w_in gradient
        internal.t_1 .= (internal.s .^ 2f0 .- 1f0)                    # temp variable
        internal.w_in[1:(n+1):end] .= -2f0 ./ (internal.t_1 .+ inw.c)      # modify weight matrix for on the go updating of d_in
        internal.g_in_ .= internal.w_in' * internal.g_in_                  # initialize with weighted sum of prev derivatives
        _addprev!(internal.g_in_, internal.s)
        internal.g_in_ .*= -2f0 .* internal.t_1 .* internal.expZ  # part 2 of on the go updating
        internal.g_in_ ./= internal.dnom .^ 2                        # final derivative of current stage
        # calculate new scores and reset weights
        internal.s .= internal.num ./ internal.dnom
        internal.w_in[1:(n+1):end] .= 0.0f0
    end
    internal.s[end] = inw.bias_out
    return nothing
end



function back_prop!(
        internal::_internal,
        y::Array{Float32,1},
        inw::InfluenceNetwork)::Nothing
    max_::Float32 = maximum(internal.out)
    internal.out .= exp.(internal.out .- max_)
    sum_ = sum(internal.out)
    internal.out ./= sum_
    internal.g_loss_ .= reshape(kron(internal.out, internal.out), inw.m, inw.m)
    internal.g_loss_[1:inw.m+1:inw.m^2] .= (internal.out) .* (internal.out .- 1f0)#(1f0 .- internal.out)
    internal.g_loss_ .*= (internal.out .- y)
    internal.g_loss .= sum(internal.g_loss_, dims = 1)[:]
    internal.g_loss .*= 2f0*inw.beta
    internal.g_init' .*= internal.g_loss
    internal.g_in' .*= internal.g_loss
    internal.g_out' .*= internal.g_loss
    return nothing
end



function predict(
        dInw::DecisionInfluenceNetwork;
        X::Array{Int, 2} = Array{Int, 2}(undef, 0, 0),
        Y::Array{Int, 1} = Array{Int, 1}(undef, 0))::_predictor
    if size(X)[1] == 0
        data = dInw.data
    else    
        data = _data(X, Y)
    end
    predictor = _predictor(dInw.inw, data)
    predict!(predictor, data, dInw.inw)
    return predictor
end

function predict!(
        predictor::_predictor, 
        data::_data, 
        inw::InfluenceNetwork)::Nothing
    predictor.S .= 0f0
    predictor.Z .= 0f0
    for i in 1:data.n
        predictor.select .= data.X[i,:].+inw.index
        predictor.Z .= inw.w_init[predictor.select, :]
        predictor.S .= _logit.(predictor.Z, _expZ.(predictor.Z)).*2f0.-1f0
        for t in 1:inw.rounds
            predictor.S[inw.n,:] .= inw.bias_in
            for k in 1:inw.m 
                predictor.Z[:,k] .= inw.w_in[predictor.select, predictor.select, k]' * predictor.S[:,k] 
            end
            predictor.S .= 
                _num.(predictor.S, predictor.Z, _expZ.(predictor.Z), inw.c)./
                _dnom.(predictor.S, predictor.Z, _expZ.(predictor.Z), inw.c)
        end
        predictor.S[inw.n,:] .= inw.bias_out
        predictor.out[i,:] .= diag(inw.w_out[predictor.select, :]' * predictor.S)
        max_::Float32 = maximum(predictor.out[i,:])
        predictor.out[i,:] .= exp.(predictor.out[i,:] .- max_)
        sum_::Float32 = sum(predictor.out[i,:])
        predictor.out[i,:] ./= sum_
    end
    return nothing
end


function _expZ(z::Float32)::Float32
    if z > 0f0
        return exp(-z)
    else
        return exp(z)
    end
end


function _dnom(x::Float32, z::Float32, expZ::Float32, c::Float32)::Float32 
    if z > 0f0
        return (1f0+x)+(1f0-x)*expZ+c
    else
        return (1f0+x)*expZ+(1f0-x)+c
    end
end
    
function _num(x::Float32, z::Float32, expZ::Float32, c::Float32)::Float32 
    if(z > 0f0)
        return (1f0+x)-(1f0-x)*expZ+c
    else
        return (1f0+x)*expZ-(1f0-x)+c
    end
end
    
function init_t_1_(z::Float32, expZ::Float32, num::Float32, dnom::Float32, c::Float32)::Float32
    if z > 0f0
        return (1f0+expZ)*dnom-(1f0-expZ)*num
    else
        return (1f0+expZ)*dnom+(1f0-expZ)*num
    end
end

function init_t_2_(x::Float32, z::Float32, expZ::Float32, num::Float32, dnom::Float32, c::Float32)::Float32
    if z > 0f0
        return (1f0-x)*expZ*(dnom-num)+c
    else
        return (1f0+x)*expZ*(dnom-num)+c
    end
end


function _logit(z::Float32, expZ::Float32)::Float32
    if z > 0.0f0
        return 1.0f0/(1.0f0+expZ)
    else
        return expZ/(1.0f0+expZ)
    end
end


function _addprev!(d_in::Array{Float32,2}, x::Array{Float32,1})::Nothing # updates derivative matrix by adding previous activation scores for some entries... check the math in technical report
    n = size(d_in)[1]
    d_in[collect(Iterators.flatten([i:n^2+1:n^3 for i in 1:n:n^2]))] .+= repeat(x, inner = n)
    return nothing
end  





