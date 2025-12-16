"""
Hamilton-Jacobi-Bellman (HJB) PDE Solver for Optimal Trading
=============================================================

이 Julia 모듈은 HJB 방정식을 풀어 최적의 레버리지 및 리밸런싱 전략을 계산합니다.

수학적 배경:
- HJB 방정식: ∂V/∂t + max_u [L^u V] = 0
- 여기서 V는 가치 함수, u는 제어 변수 (레버리지, 포지션)
- L^u는 infinitesimal generator

목적:
1. 리스크 조정 수익률 최대화
2. 동적 레버리지 결정
3. 최적 리밸런싱 타이밍

참고 문헌:
- Merton (1969, 1971): Optimal consumption and portfolio rules
- Almgren & Chriss (2001): Optimal execution of portfolio transactions
"""

using LinearAlgebra
using SparseArrays
using JSON

"""
HJB Solver 설정 구조체
"""
struct HJBConfig
    # 시간 그리드
    T::Float64              # 총 시간 (일 단위)
    dt::Float64             # 시간 간격
    
    # 상태 공간 그리드
    wealth_min::Float64     # 최소 자산
    wealth_max::Float64     # 최대 자산
    n_wealth::Int           # 자산 그리드 수
    
    # 시장 파라미터
    mu::Float64             # 기대 수익률 (연율)
    sigma::Float64          # 변동성 (연율)
    risk_free_rate::Float64 # 무위험 이자율
    
    # 제어 파라미터
    leverage_min::Float64   # 최소 레버리지
    leverage_max::Float64   # 최대 레버리지
    n_leverage::Int         # 레버리지 그리드 수
    
    # 비용 파라미터
    transaction_cost::Float64  # 거래 비용 (%)
    impact_coef::Float64       # Market impact 계수
    
    # 리스크 회피 파라미터
    gamma::Float64          # 상대적 리스크 회피 계수 (CRRA)
end

"""
기본 설정 생성
"""
function default_config()::HJBConfig
    return HJBConfig(
        # 시간
        T = 252.0,           # 1년
        dt = 1.0,            # 1일
        
        # 자산 그리드
        wealth_min = 0.5,
        wealth_max = 2.0,
        n_wealth = 100,
        
        # 시장
        mu = 0.10,           # 10% 연율 수익률
        sigma = 0.20,        # 20% 연율 변동성
        risk_free_rate = 0.02,
        
        # 제어
        leverage_min = 0.0,
        leverage_max = 3.0,
        n_leverage = 31,
        
        # 비용
        transaction_cost = 0.001,  # 0.1%
        impact_coef = 0.1,
        
        # 리스크 회피
        gamma = 2.0          # 중간 수준 리스크 회피
    )
end

"""
CRRA 효용 함수 (Constant Relative Risk Aversion)
U(W) = W^(1-γ) / (1-γ)
"""
function utility(W::Float64, gamma::Float64)::Float64
    if gamma ≈ 1.0
        return log(max(W, 1e-10))
    else
        return (max(W, 1e-10)^(1 - gamma)) / (1 - gamma)
    end
end

"""
HJB 방정식의 Hamiltonian 계산

H = max_l [ μ*l*W*∂V/∂W + 0.5*σ²*l²*W²*∂²V/∂W² - c(l)*W ]

여기서:
- l: 레버리지
- W: 자산
- c(l): 거래 비용 함수
"""
function compute_hamiltonian(
    W::Float64,
    V::Float64,
    dV_dW::Float64,
    d2V_dW2::Float64,
    leverage::Float64,
    config::HJBConfig
)::Float64
    
    # 포트폴리오 드리프트
    drift = config.mu * leverage * W * dV_dW
    
    # 포트폴리오 확산 (변동성)
    diffusion = 0.5 * (config.sigma * leverage * W)^2 * d2V_dW2
    
    # 거래 비용 (레버리지 변경 시)
    # 간단한 모델: cost = tc * |Δl| * W
    # 여기서는 레버리지 유지 비용으로 근사
    cost = config.transaction_cost * abs(leverage) * W
    
    # Hamiltonian
    H = drift + diffusion - cost
    
    return H
end

"""
유한 차분법으로 미분 계산
"""
function compute_derivatives(
    V::Vector{Float64},
    W_grid::Vector{Float64}
)::Tuple{Vector{Float64}, Vector{Float64}}
    
    n = length(V)
    dV = zeros(n)
    d2V = zeros(n)
    
    # 중앙 차분 (interior points)
    for i in 2:(n-1)
        dW = W_grid[i+1] - W_grid[i-1]
        dV[i] = (V[i+1] - V[i-1]) / dW
        
        dW_plus = W_grid[i+1] - W_grid[i]
        dW_minus = W_grid[i] - W_grid[i-1]
        d2V[i] = 2 * (V[i+1] - V[i]) / (dW_plus * (dW_plus + dW_minus)) -
                 2 * (V[i] - V[i-1]) / (dW_minus * (dW_plus + dW_minus))
    end
    
    # 경계 조건 (forward/backward difference)
    dV[1] = (V[2] - V[1]) / (W_grid[2] - W_grid[1])
    dV[n] = (V[n] - V[n-1]) / (W_grid[n] - W_grid[n-1])
    
    d2V[1] = d2V[2]
    d2V[n] = d2V[n-1]
    
    return dV, d2V
end

"""
최적 레버리지 계산 (각 자산 수준에서)
"""
function compute_optimal_leverage(
    W::Float64,
    V::Float64,
    dV_dW::Float64,
    d2V_dW2::Float64,
    leverage_grid::Vector{Float64},
    config::HJBConfig
)::Tuple{Float64, Float64}
    
    best_leverage = 0.0
    best_H = -Inf
    
    for l in leverage_grid
        H = compute_hamiltonian(W, V, dV_dW, d2V_dW2, l, config)
        
        if H > best_H
            best_H = H
            best_leverage = l
        end
    end
    
    return best_leverage, best_H
end

"""
HJB 방정식 풀이 (Backward in time)
"""
function solve_hjb(config::HJBConfig)::Dict{String, Any}
    
    # 그리드 생성
    W_grid = range(config.wealth_min, config.wealth_max, length=config.n_wealth)
    leverage_grid = range(config.leverage_min, config.leverage_max, length=config.n_leverage)
    n_steps = Int(ceil(config.T / config.dt))
    
    # 가치 함수 초기화 (terminal condition)
    V = [utility(W, config.gamma) for W in W_grid]
    
    # 최적 레버리지 저장
    optimal_leverage = zeros(config.n_wealth, n_steps)
    
    # Backward iteration
    for t_idx in n_steps:-1:1
        t = t_idx * config.dt
        
        # 미분 계산
        dV, d2V = compute_derivatives(V, collect(W_grid))
        
        # 각 자산 수준에서 최적 제어 계산
        V_new = zeros(config.n_wealth)
        
        for (i, W) in enumerate(W_grid)
            l_opt, H_opt = compute_optimal_leverage(
                W, V[i], dV[i], d2V[i], collect(leverage_grid), config
            )
            
            optimal_leverage[i, t_idx] = l_opt
            
            # 가치 함수 업데이트 (implicit scheme)
            # V^{n+1} = V^n + dt * H
            V_new[i] = V[i] + config.dt * H_opt
        end
        
        V = V_new
    end
    
    # 결과 반환
    result = Dict(
        "wealth_grid" => collect(W_grid),
        "leverage_grid" => collect(leverage_grid),
        "value_function" => V,
        "optimal_leverage" => optimal_leverage,
        "config" => Dict(
            "T" => config.T,
            "dt" => config.dt,
            "mu" => config.mu,
            "sigma" => config.sigma,
            "gamma" => config.gamma
        )
    )
    
    return result
end

"""
현재 자산 수준에 대한 최적 레버리지 조회
"""
function get_optimal_leverage_for_wealth(
    result::Dict{String, Any},
    current_wealth::Float64,
    time_step::Int = 1
)::Float64
    
    W_grid = result["wealth_grid"]
    opt_lev = result["optimal_leverage"]
    
    # 선형 보간
    if current_wealth <= W_grid[1]
        return opt_lev[1, time_step]
    elseif current_wealth >= W_grid[end]
        return opt_lev[end, time_step]
    else
        # 가장 가까운 두 점 찾기
        idx = searchsortedfirst(W_grid, current_wealth)
        if idx == 1
            idx = 2
        end
        
        W_low = W_grid[idx-1]
        W_high = W_grid[idx]
        
        # 선형 보간
        weight = (current_wealth - W_low) / (W_high - W_low)
        lev_low = opt_lev[idx-1, time_step]
        lev_high = opt_lev[idx, time_step]
        
        return lev_low + weight * (lev_high - lev_low)
    end
end

"""
Python에서 호출할 메인 함수
"""
function solve_and_export(
    mu::Float64 = 0.10,
    sigma::Float64 = 0.20,
    gamma::Float64 = 2.0,
    current_wealth::Float64 = 1.0
)::String
    
    # 설정 생성
    config = HJBConfig(
        T = 252.0,
        dt = 1.0,
        wealth_min = 0.5,
        wealth_max = 2.0,
        n_wealth = 50,  # 계산 속도를 위해 축소
        mu = mu,
        sigma = sigma,
        risk_free_rate = 0.02,
        leverage_min = 0.0,
        leverage_max = 3.0,
        n_leverage = 31,
        transaction_cost = 0.001,
        impact_coef = 0.1,
        gamma = gamma
    )
    
    # HJB 풀이
    result = solve_hjb(config)
    
    # 현재 자산에 대한 최적 레버리지
    optimal_lev = get_optimal_leverage_for_wealth(result, current_wealth, 1)
    
    # 결과 요약
    summary = Dict(
        "optimal_leverage" => optimal_lev,
        "current_wealth" => current_wealth,
        "expected_return" => mu,
        "volatility" => sigma,
        "risk_aversion" => gamma,
        "wealth_grid_min" => result["wealth_grid"][1],
        "wealth_grid_max" => result["wealth_grid"][end],
        "avg_leverage" => mean(result["optimal_leverage"][:, 1])
    )
    
    return JSON.json(summary)
end

# 테스트 실행 (Julia에서 직접 실행 시)
if abspath(PROGRAM_FILE) == @__FILE__
    println("HJB Solver 테스트 실행...")
    result_json = solve_and_export(0.12, 0.25, 2.0, 1.0)
    println(result_json)
end

