using LinearAlgebra


struct InfluenceNetwork 
    n::Int
    n_::Int
    m::Int
    rounds::Int
    bias_in::Float32
    bias_out::Float32
    beta::Float32
    c::Float32
    scale::Float32
    index::Array{Int, 1}
    w_init::Array{Float32, 2}
    w_in::Array{Float32, 3}
    w_out::Array{Float32, 2}
    free::Array{Float32, 3}
    function InfluenceNetwork(
            n::Int,
            m::Int;
            rounds::Int = 4,
            bias_in::Bool = true,
            bias_out::Bool = true,
            beta::Float32 = 1f0,
            c::Float32 = 1f-9,
            scale::Float32 = 1f-2,
            index::Array{Int, 1} = Array{Int, 1}(undef, 0),
            w_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            w_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
            w_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            free::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0))
        scale *= 2.0f0
        if size(index)[1] == 0
            index = collect(1:2:2*n)
        end
        n_::Int = index[end]
        if size(free)[1] != n_
            free = reshape(repeat(Float32.(.!(Array{Bool, 2}(I, n_, n_))), m)', n_, n_, m)
            free[:,n_,:] .= 0f0
            #free = Float32.(.!(Array{Bool, 2}(I, n_, n_, m)))
            #free[:,size(free)[1]] .= 0.0f0
        end
        if size(w_init)[1] != n_
            w_init = (rand(Float32, n_, m) .- 0.5f0) .* scale 
        end
        if size(w_in)[1] != n_
            w_in = (rand(Float32, n_, n_, m) .- 0.5f0) .* free .* scale
        end
        if size(w_out)[1] != n_
            w_out = (rand(Float32, n_, m) .- 0.5f0) .* scale 
        end
        return new(
            n,
            n_,
            m,
            rounds,
            Float32(bias_in),
            Float32(bias_out),
            beta,
            c,
            scale,
            index,
            w_init,
            w_in,
            w_out,
            free)
    end
end


struct _data
    X::Array{Int, 2}
    Y::Array{Float32, 2}
    n::Int
    m::Int
    function _data(
            X::Array{Int, 2}, 
            Y::Array{Int, 1};
            binary::Bool = false,
            shuffle_::Bool = false)
        n::Int = size(X)[1]
        m::Int = size(X)[2]
        nCases::Int = binary ? 1 : length(unique(Y))
        order::Array{Int, 1} = shuffle_ ? shuffle(1:n) : collect(1:n)
        X_::Array{Int, 2} = zeros(Int, (n, m+1))
        X_[1:n,1:m] .= X[order,:]
        if length(Y) > 0
            Y_ = zeros(Float32, n, nCases)
            for i in 1:n
                Y_[i,Y[order[i]]+1] = 1f0
            end
        else
            Y_ = Array{Float32, 2}(undef,0,0)
        end
        return new(X_, Y_, n, m)
    end
end


struct DecisionInfluenceNetwork
    inw::InfluenceNetwork
    data::_data
    function DecisionInfluenceNetwork(
            X::Array{Int, 2}, 
            Y::Array{Int, 1};
            binary::Bool = false,
            shuffle_::Bool = false,
            rounds::Int = 4,
            bias_in::Bool = true,
            bias_out::Bool = true,
            c::Float32 = 1f-9,
            scale::Float32 = 1f-2,
            w_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            w_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
            w_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            free::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0))
        data = _data(
            X,
            Y,
            binary = binary,
            shuffle_ = shuffle_)
        index::Array{Int, 1} = ones(Int, data.m+1)
        for i in 1:data.m
            index[i+1] = index[i] + length(unique(data.X[:,i]))
        end
        inw = InfluenceNetwork(
            data.m+1,
            size(data.Y)[2],
            rounds = rounds,
            bias_in = bias_in,
            bias_out = bias_out,
            c = c,
            scale = scale,
            index = index,
            w_init = w_init,
            w_in = w_in,
            w_out = w_out,
            free = free)
        return new(inw, data)
    end
end



struct _param # values of training parameters
    minIter::Int
    maxIter::Int # max number of training iterations
    lr::Float32
    lr_fact::Float32 # learning rate
    mom::Float32 # momentum term
    thresh::Float32 # critical gradient value, break training upon falling below
    batchSize::Int
    testSize::Int
    feedback::Int
    function _param(; 
            minIter::Int = 500,
            maxIter::Int = 10000,
            lr::Float32 = 1f-2,
            lr_fact::Float32 = 1f0,            
            mom::Float32 = 25f-2, 
            thresh::Float32 = 1f-6,
            batchSize::Int = -1,
            testSize::Int = 0,
            feedback::Int = 10)
        return new(minIter, maxIter, lr, lr_fact, mom, thresh, batchSize, testSize, feedback)
    end
end


struct _trainer
    g_init::Array{Float32, 2}
    g_in::Array{Float32, 3}
    g_out::Array{Float32, 2}
    mom_init::Array{Float32, 2}
    mom_in::Array{Float32, 3}
    mom_out::Array{Float32, 2}
    upd_count_in::Array{Float32, 2}
    upd_count_out::Array{Float32, 1}
    update_in::BitArray{2}
    update_out::BitArray{1}
    select::Array{Int, 1}
    function _trainer(
            inw::InfluenceNetwork;
            g_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            g_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
            g_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            mom_init::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0),
            mom_in::Array{Float32, 3} = Array{Float32, 3}(undef, 0, 0, 0),
            mom_out::Array{Float32, 2} = Array{Float32, 2}(undef, 0, 0))
        if size(g_init)[1] != inw.n_
            g_init = Array{Float32, 2}(undef, inw.n_, inw.m)
        end
        if size(g_in)[1] != inw.n_
            g_in = Array{Float32, 3}(undef, inw.n_, inw.n_, inw.m)
        end
        if size(g_out)[1] != inw.n_
            g_out = Array{Float32, 2}(undef, inw.n_, inw.m)
        end
        if size(mom_init)[1] != inw.n_
            mom_init = zeros(Float32, inw.n_, inw.m)
        end
        if size(mom_in)[1] != inw.n_
            mom_in = zeros(Float32, inw.n_, inw.n_, inw.m)
        end
        if size(mom_out)[1] != inw.n_
            mom_out = zeros(Float32, inw.n_, inw.m)
        end
        upd_count_in = Array{Float32, 2}(undef, inw.n_, inw.n_)
        upd_count_out = Array{Float32, 1}(undef, inw.n_)
        update_in = BitArray{2}(undef, inw.n_, inw.n_)
        update_out = BitArray{1}(undef, inw.n_)
        select::Array{Int, 1} = Array{Int, 1}(undef, inw.n)
        return new(g_init, g_in, g_out, mom_init, mom_in, mom_out, upd_count_in, upd_count_out, update_in, update_out, select)
    end
end 


struct _internal
    out::Array{Float32, 1}
    w_init::Array{Float32, 1} 
    w_in::Array{Float32, 2} 
    w_out::Array{Float32, 1}
    s::Array{Float32, 1}
    z::Array{Float32, 1}
    expZ::Array{Float32, 1}
    t_1::Array{Float32, 1}
    t_2::Array{Float32, 1}
    num::Array{Float32, 1}
    dnom::Array{Float32, 1}
    g_init::Array{Float32, 2}
    g_in::Array{Float32, 2}
    g_out::Array{Float32, 2}
    g_loss::Array{Float32, 1}
    g_init_::Array{Float32, 2}
    g_in_::Array{Float32, 2}
    g_loss_::Array{Float32, 2}
    function _internal(inw::InfluenceNetwork)
        out::Array{Float32, 1} = Array{Float32, 1}(undef, inw.m)
        w_init::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        w_in::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.n)
        w_out::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        s::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        z::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        expZ::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        t_1::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        t_2::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        num::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        dnom::Array{Float32, 1} = Array{Float32, 1}(undef, inw.n)
        g_init::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.m)
        g_in::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n^2, inw.m)
        g_out::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.m)
        g_loss::Array{Float32, 1} = Array{Float32, 1}(undef, inw.m)
        g_init_::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.n)
        g_in_::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.n^2)
        g_loss_::Array{Float32, 2} = Array{Float32, 2}(undef, inw.m, inw.m)
        return new(out, w_init, w_in, w_out, s, z, expZ, t_1, t_2, num, dnom, g_init, g_in, g_out, g_loss, g_init_, g_in_, g_loss_)
    end
end


struct _predictor
    out::Array{Float32, 2}
    select::Array{Int, 1}
    S::Array{Float32, 2}
    Z::Array{Float32, 2}
    function _predictor(inw::InfluenceNetwork, data::_data)
        out::Array{Float32, 2} = Array{Float32, 2}(undef, data.n, inw.m)
        select::Array{Int, 1} = Array{Int, 1}(undef, inw.n)
        S::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.m)
        Z::Array{Float32, 2} = Array{Float32, 2}(undef, inw.n, inw.m)
        return new(out, select, S, Z)
    end
end
