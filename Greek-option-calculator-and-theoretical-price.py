import math
import numpy as np
from scipy.stats import norm

class OptionCalculator:
    """
    Профессиональный калькулятор стоимости опционов с поддержкой всех методик
    """
    
    def __init__(self, S, K, T, r, sigma, option_type='call', 
                 asset_type='futures', dividend_yield=0, n_steps=1000, 
                 n_simulations=100000, dividends=None):
        """
        Инициализация параметров опциона
        """
        self.S = S
        self.K = K
        self.T = T / 365.0  # Конвертируем дни в годы
        self.r = r
        self.sigma = sigma  # Волатильность в % (0.20 для 20%)
        self.option_type = option_type.lower()
        self.asset_type = asset_type.lower()
        self.dividend_yield = dividend_yield
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.dividends = dividends or []
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        if self.asset_type not in ['futures', 'stock', 'currency', 'index']:
            raise ValueError("Asset type must be 'futures', 'stock', 'currency', or 'index'")
    
    def show_formula_calculation(self, formula_name, steps):
        """Показывает расчет формулы по шагам"""
        print(f"\n📊 РАСЧЕТ {formula_name}:")
        print("=" * 50)
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")
        print("=" * 50)
    
    def calculate_all_greeks(self, model='black_scholes_moex'):
        """Расчет всех греков для выбранной модели"""
        try:
            if model == 'black_scholes_moex':
                return self.calculate_greeks_moex()
            elif model == 'black_scholes_classic':
                return self.calculate_greeks_classic()
            elif model == 'bachelier':
                return self.calculate_greeks_bachelier()
            else:
                return None
        except Exception as e:
            return f"Ошибка расчета греков: {e}"
    
    def calculate_greeks_moex(self):
        """Расчет греков по методике Московской Биржи для фьючерсов"""
        if self.sigma <= 0 or self.T <= 0:
            return None
            
        d1 = (math.log(self.S / self.K) + (self.sigma**2) * self.T / 2) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        
        # Основные греки
        if self.option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        gamma = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T)) * 100
        vega = self.S * norm.pdf(d1) * math.sqrt(self.T) / 100
        
        # Тета (за 1 день)
        theta_per_day = - (self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T)) / 365
        
        # Ро (для фьючерсов = 0)
        rho = 0
        
        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'vega': round(vega, 6),
            'theta': round(theta_per_day, 6),
            'rho': round(rho, 6)
        }
    
    def calculate_greeks_classic(self):
        """Расчет греков для классической модели Б-Ш"""
        if self.sigma <= 0 or self.T <= 0:
            return None
            
        d1 = (math.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        
        if self.option_type == 'call':
            delta = norm.cdf(d1)
            rho = self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            rho = -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-d2) / 100
            
        gamma = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T)) * 100
        vega = self.S * norm.pdf(d1) * math.sqrt(self.T) / 100
        
        # Тета (за 1 день)
        if self.option_type == 'call':
            theta_per_day = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T)) - 
                            self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)) / 365
        else:
            theta_per_day = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * math.sqrt(self.T)) + 
                            self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)) / 365
        
        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'vega': round(vega, 6),
            'theta': round(theta_per_day, 6),
            'rho': round(rho, 6)
        }
    
    def calculate_greeks_bachelier(self):
        """Расчет греков для модели Башелье"""
        if self.sigma <= 0 or self.T <= 0:
            return None
            
        # Для модели Башелье используем абсолютную волатильность
        sigma_abs = self.sigma * self.S
        
        d = (self.S - self.K) / (sigma_abs * math.sqrt(self.T))
        
        if self.option_type == 'call':
            delta = norm.cdf(d)
        else:
            delta = norm.cdf(d) - 1
            
        gamma = norm.pdf(d) / (sigma_abs * math.sqrt(self.T)) * 100
        vega = math.sqrt(self.T) * norm.pdf(d) / 100
        theta_per_day = - (sigma_abs * norm.pdf(d)) / (2 * math.sqrt(self.T)) / 365
        rho = 0
        
        return {
            'delta': round(delta, 6),
            'gamma': round(gamma, 6),
            'vega': round(vega, 6),
            'theta': round(theta_per_day, 6),
            'rho': round(rho, 6)
        }
    
    def black_scholes_moex(self):
        """Модель Блэка-Шоулза по методике МосБиржи (БЕЗ ставки для фьючерсов)"""
        try:
            if self.sigma <= 0:
                if self.option_type == 'call':
                    return max(self.S - self.K, 0)
                else:
                    return max(self.K - self.S, 0)
            
            # Показываем расчет d1 и d2
            log_sk = math.log(self.S / self.K)
            sigma_sq_t = (self.sigma**2) * self.T / 2
            denominator = self.sigma * math.sqrt(self.T)
            
            d1 = (log_sk + sigma_sq_t) / denominator
            d2 = d1 - self.sigma * math.sqrt(self.T)
            
            steps = [
                f"d1 = ln(S/K) + (σ² * T/2) / (σ * √T)",
                f"d1 = ln({self.S}/{self.K}) + ({self.sigma:.6f}² * {self.T:.6f}/2) / ({self.sigma:.6f} * √{self.T:.6f})",
                f"d1 = {log_sk:.6f} + {sigma_sq_t:.6f} / {denominator:.6f}",
                f"d1 = {d1:.6f}",
                f"d2 = d1 - σ * √T = {d1:.6f} - {self.sigma:.6f} * √{self.T:.6f}",
                f"d2 = {d2:.6f}",
                f"N(d1) = {norm.cdf(d1):.6f}, N(d2) = {norm.cdf(d2):.6f}"
            ]
            
            if self.option_type == 'call':
                price = self.S * norm.cdf(d1) - self.K * norm.cdf(d2)
                steps.extend([
                    f"Цена CALL = S * N(d1) - K * N(d2)",
                    f"Цена CALL = {self.S} * {norm.cdf(d1):.6f} - {self.K} * {norm.cdf(d2):.6f}",
                    f"Цена CALL = {self.S * norm.cdf(d1):.6f} - {self.K * norm.cdf(d2):.6f}",
                    f"Цена CALL = {price:.6f}"
                ])
            else:
                price = self.K * norm.cdf(-d2) - self.S * norm.cdf(-d1)
                steps.extend([
                    f"Цена PUT = K * N(-d2) - S * N(-d1)",
                    f"Цена PUT = {self.K} * {norm.cdf(-d2):.6f} - {self.S} * {norm.cdf(-d1):.6f}",
                    f"Цена PUT = {self.K * norm.cdf(-d2):.6f} - {self.S * norm.cdf(-d1):.6f}",
                    f"Цена PUT = {price:.6f}"
                ])
            
            self.show_formula_calculation("BLACK-SCHOLES MOEX", steps)
            
            return round(max(price, 0), 4)
            
        except Exception as e:
            return f"Ошибка в Black-Scholes MOEX: {e}"
    
    def black_scholes_classic(self):
        """Классическая модель Блэка-Шоулза (СО ставкой)"""
        try:
            if self.sigma <= 0:
                if self.option_type == 'call':
                    return max(self.S - self.K * math.exp(-self.r * self.T), 0)
                else:
                    return max(self.K * math.exp(-self.r * self.T) - self.S, 0)
            
            # Показываем расчет d1 и d2
            log_sk = math.log(self.S / self.K)
            r_sigma_sq = (self.r + self.sigma**2 / 2) * self.T
            denominator = self.sigma * math.sqrt(self.T)
            
            d1 = (log_sk + r_sigma_sq) / denominator
            d2 = d1 - self.sigma * math.sqrt(self.T)
            
            discount_factor = math.exp(-self.r * self.T)
            
            steps = [
                f"d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)",
                f"d1 = [ln({self.S}/{self.K}) + ({self.r:.6f} + {self.sigma:.6f}²/2) * {self.T:.6f}] / ({self.sigma:.6f} * √{self.T:.6f})",
                f"d1 = [{log_sk:.6f} + {r_sigma_sq:.6f}] / {denominator:.6f}",
                f"d1 = {d1:.6f}",
                f"d2 = d1 - σ * √T = {d1:.6f} - {self.sigma:.6f} * √{self.T:.6f}",
                f"d2 = {d2:.6f}",
                f"Коэффициент дисконтирования e^(-rT) = e^(-{self.r:.6f}*{self.T:.6f}) = {discount_factor:.6f}",
                f"N(d1) = {norm.cdf(d1):.6f}, N(d2) = {norm.cdf(d2):.6f}"
            ]
            
            if self.option_type == 'call':
                price = self.S * norm.cdf(d1) - self.K * discount_factor * norm.cdf(d2)
                steps.extend([
                    f"Цена CALL = S * N(d1) - K * e^(-rT) * N(d2)",
                    f"Цена CALL = {self.S} * {norm.cdf(d1):.6f} - {self.K} * {discount_factor:.6f} * {norm.cdf(d2):.6f}",
                    f"Цена CALL = {self.S * norm.cdf(d1):.6f} - {self.K * discount_factor * norm.cdf(d2):.6f}",
                    f"Цена CALL = {price:.6f}"
                ])
            else:
                price = self.K * discount_factor * norm.cdf(-d2) - self.S * norm.cdf(-d1)
                steps.extend([
                    f"Цена PUT = K * e^(-rT) * N(-d2) - S * N(-d1)",
                    f"Цена PUT = {self.K} * {discount_factor:.6f} * {norm.cdf(-d2):.6f} - {self.S} * {norm.cdf(-d1):.6f}",
                    f"Цена PUT = {self.K * discount_factor * norm.cdf(-d2):.6f} - {self.S * norm.cdf(-d1):.6f}",
                    f"Цена PUT = {price:.6f}"
                ])
            
            self.show_formula_calculation("CLASSIC BLACK-SCHOLES", steps)
            
            return round(max(price, 0), 4)
        except Exception as e:
            return f"Ошибка в классическом Black-Scholes: {e}"
    
    def bachelier_model(self):
        """Модель Башелье (альтернативная модель)"""
        try:
            if self.sigma <= 0:
                if self.option_type == 'call':
                    return max(self.S - self.K, 0)
                else:
                    return max(self.K - self.S, 0)
            
            # Для модели Башелье используем абсолютную волатильность
            sigma_abs = self.sigma * self.S
            
            d = (self.S - self.K) / (sigma_abs * math.sqrt(self.T))
            
            steps = [
                f"Абсолютная волатильность σ_abs = σ * S = {self.sigma:.6f} * {self.S} = {sigma_abs:.6f}",
                f"d = (S - K) / (σ_abs * √T)",
                f"d = ({self.S} - {self.K}) / ({sigma_abs:.6f} * √{self.T:.6f})",
                f"d = {self.S - self.K:.6f} / {sigma_abs * math.sqrt(self.T):.6f}",
                f"d = {d:.6f}",
                f"N(d) = {norm.cdf(d):.6f}, φ(d) = {norm.pdf(d):.6f}"
            ]
            
            if self.option_type == 'call':
                price = ((self.S - self.K) * norm.cdf(d) + 
                        sigma_abs * math.sqrt(self.T) * norm.pdf(d))
                steps.extend([
                    f"Цена CALL = (S - K) * N(d) + σ_abs * √T * φ(d)",
                    f"Цена CALL = ({self.S} - {self.K}) * {norm.cdf(d):.6f} + {sigma_abs:.6f} * √{self.T:.6f} * {norm.pdf(d):.6f}",
                    f"Цена CALL = {self.S - self.K:.6f} * {norm.cdf(d):.6f} + {sigma_abs * math.sqrt(self.T):.6f} * {norm.pdf(d):.6f}",
                    f"Цена CALL = {(self.S - self.K) * norm.cdf(d):.6f} + {sigma_abs * math.sqrt(self.T) * norm.pdf(d):.6f}",
                    f"Цена CALL = {price:.6f}"
                ])
            else:
                price = ((self.K - self.S) * norm.cdf(-d) + 
                        sigma_abs * math.sqrt(self.T) * norm.pdf(d))
                steps.extend([
                    f"Цена PUT = (K - S) * N(-d) + σ_abs * √T * φ(d)",
                    f"Цена PUT = ({self.K} - {self.S}) * {norm.cdf(-d):.6f} + {sigma_abs:.6f} * √{self.T:.6f} * {norm.pdf(d):.6f}",
                    f"Цена PUT = {self.K - self.S:.6f} * {norm.cdf(-d):.6f} + {sigma_abs * math.sqrt(self.T):.6f} * {norm.pdf(d):.6f}",
                    f"Цена PUT = {(self.K - self.S) * norm.cdf(-d):.6f} + {sigma_abs * math.sqrt(self.T) * norm.pdf(d):.6f}",
                    f"Цена PUT = {price:.6f}"
                ])
            
            self.show_formula_calculation("BACHELIER MODEL", steps)
            
            return round(max(price, 0), 4)
        except Exception as e:
            return f"Ошибка в модели Башелье: {e}"
    
    def binomial_tree(self, use_r=True):
        """Биномиальная модель с опцией использования ставки"""
        try:
            if self.T <= 0:
                if self.option_type == 'call':
                    return max(self.S - self.K, 0)
                else:
                    return max(self.K - self.S, 0)
            
            dt = self.T / self.n_steps
            u = math.exp(self.sigma * math.sqrt(dt))
            d = 1 / u
            
            # Используем или не используем ставку
            if use_r:
                p = (math.exp(self.r * dt) - d) / (u - d)
                method_name = "BINOMIAL TREE WITH R"
            else:
                p = (1 - d) / (u - d)
                method_name = "BINOMIAL TREE WITHOUT R"
            
            steps = [
                f"Δt = T/n = {self.T:.6f}/{self.n_steps} = {dt:.6f}",
                f"u = e^(σ√Δt) = e^({self.sigma:.6f}*√{dt:.6f}) = {u:.6f}",
                f"d = 1/u = {d:.6f}",
                f"p = {'(e^(rΔt) - d)/(u - d)' if use_r else '(1 - d)/(u - d)'}",
                f"p = {'(e^(' + str(self.r) + '*' + str(dt) + ') - ' + str(d) + ')/(' + str(u) + ' - ' + str(d) + ')' if use_r else '(1 - ' + str(d) + ')/(' + str(u) + ' - ' + str(d) + ')'}",
                f"p = {p:.6f}"
            ]
            
            # Создаем дерево цен на конечном шаге
            prices = np.zeros(self.n_steps + 1)
            for i in range(self.n_steps + 1):
                prices[i] = self.S * (u ** (self.n_steps - i)) * (d ** i)
            
            # Вычисляем значения опциона на конечных узлах
            if self.option_type == 'call':
                values = np.maximum(prices - self.K, 0)
            else:
                values = np.maximum(self.K - prices, 0)
            
            # Обратный проход по дереву
            for step in range(self.n_steps - 1, -1, -1):
                if use_r:
                    discount = math.exp(-self.r * dt)
                    values = discount * (p * values[:-1] + (1 - p) * values[1:])
                else:
                    values = p * values[:-1] + (1 - p) * values[1:]
            
            steps.extend([
                f"Количество шагов: {self.n_steps}",
                f"Конечная цена: {values[0]:.6f}",
                f"Метод: {'с учетом ставки' if use_r else 'без учета ставки'}"
            ])
            
            self.show_formula_calculation(method_name, steps)
            
            return round(values[0], 4)
        except Exception as e:
            return f"Ошибка в Binomial Tree: {e}"
    
    def monte_carlo(self, use_r=True):
        """Метод Монте-Карло с опцией использования ставки"""
        try:
            if self.T <= 0:
                if self.option_type == 'call':
                    return max(self.S - self.K, 0)
                else:
                    return max(self.K - self.S, 0)
            
            np.random.seed(42)
            z = np.random.standard_normal(self.n_simulations)
            
            if use_r:
                drift = (self.r - 0.5 * self.sigma**2) * self.T
                method_name = "MONTE CARLO WITH R"
            else:
                drift = -0.5 * self.sigma**2 * self.T
                method_name = "MONTE CARLO WITHOUT R"
                
            ST = self.S * np.exp(drift + self.sigma * math.sqrt(self.T) * z)
            
            if self.option_type == 'call':
                payoff = np.maximum(ST - self.K, 0)
            else:
                payoff = np.maximum(self.K - ST, 0)
            
            if use_r:
                price = np.exp(-self.r * self.T) * np.mean(payoff)
            else:
                price = np.mean(payoff)
            
            steps = [
                f"Количество симуляций: {self.n_simulations}",
                f"Дрейф = {'(r - σ²/2)*T' if use_r else '(-σ²/2)*T'}",
                f"Дрейф = {'(' + str(self.r) + ' - ' + str(self.sigma**2/2) + ')*' + str(self.T) if use_r else '(-' + str(self.sigma**2/2) + ')*' + str(self.T)}",
                f"Дрейф = {drift:.6f}",
                f"Средняя цена payoff: {np.mean(payoff):.6f}",
                f"{'Дисконтированная цена: e^(-rT) * mean(payoff)' if use_r else 'Цена: mean(payoff)'}",
                f"{'e^(-' + str(self.r) + '*' + str(self.T) + ') * ' + str(np.mean(payoff)) if use_r else ''}",
                f"Результат: {price:.6f}"
            ]
            
            self.show_formula_calculation(method_name, steps)
                
            return round(price, 4)
        except Exception as e:
            return f"Ошибка в Monte Carlo: {e}"
    
    def calculate_all(self):
        """Расчет всеми методами и моделями"""
        results = {
            # Основные модели
            'MOEX_Black_Scholes': self.black_scholes_moex(),
            'Classic_Black_Scholes': self.black_scholes_classic(),
            'Bachelier_Model': self.bachelier_model(),
            
            # Биномиальные деревья
            'Binomial_Tree_With_R': self.binomial_tree(use_r=True),
            'Binomial_Tree_Without_R': self.binomial_tree(use_r=False),
            
            # Монте-Карло
            'Monte_Carlo_With_R': self.monte_carlo(use_r=True),
            'Monte_Carlo_Without_R': self.monte_carlo(use_r=False),
            
            # Греки
            'Greeks_MOEX': self.calculate_greeks_moex(),
            'Greeks_Classic': self.calculate_greeks_classic(),
            'Greeks_Bachelier': self.calculate_greeks_bachelier()
        }
        
        return results

def manual_input():
    """Ручной ввод параметров опциона"""
    print("\n=== РУЧНОЙ ВВОД ПАРАМЕТРОВ ===")
    
    S = float(input("Текущая цена актива (S): "))
    K = float(input("Цена исполнения (страйк, K): "))
    T_days = float(input("Время до экспирации в днях: "))
    option_type = input("Тип опциона (call/put): ").lower()
    iv = float(input("Волатильность (например, 0.20 для 20%): "))
    r = float(input("Безрисковая ставка (например, 0.05 для 5%): ") or "0.165")
    
    # Тип базового актива
    print("\nВыберите тип базового актива:")
    print("1 - Фьючерсы")
    print("2 - Акции")
    print("3 - Валюты")
    print("4 - Индексы")
    
    asset_choice = input("Ваш выбор (1-4): ") or "1"
    asset_types = {'1': 'futures', '2': 'stock', '3': 'currency', '4': 'index'}
    asset_type = asset_types.get(asset_choice, 'futures')
    
    # Дополнительные параметры
    n_steps = int(input("Шагов для биномиального дерева (100-1000): ") or "500")
    n_simulations = int(input("Симуляций для Монте-Карло (10000-100000): ") or "50000")
    
    return {
        'S': S,
        'K': K,
        'T_days': T_days,
        'r': r,
        'iv': iv,
        'option_type': option_type,
        'asset_type': asset_type,
        'n_steps': n_steps,
        'n_simulations': n_simulations
    }

def main():
    """Главная функция"""
    print("=== ПРОФЕССИОНАЛЬНЫЙ КАЛЬКУЛЯТОР ОПЦИОНОВ ===\n")
    
    try:
        manual_data = manual_input()
        
        # Используем ручной ввод
        S = manual_data['S']
        K = manual_data['K']
        T_days = manual_data['T_days']
        option_type = manual_data['option_type']
        iv = manual_data['iv']
        r = manual_data['r']
        asset_type = manual_data['asset_type']
        n_steps = manual_data['n_steps']
        n_simulations = manual_data['n_simulations']
    
        # Создаем калькулятор
        calculator = OptionCalculator(
            S, K, T_days, r, iv, option_type, asset_type, 
            0, n_steps, n_simulations, []
        )
        
        # Производим расчет
        results = calculator.calculate_all()
        
        # Вывод результатов
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ РАСЧЕТА")
        print("="*60)
        
        print(f"\n{'МОДЕЛЬ':<25} {'ЦЕНА':<10}")
        print("-" * 35)
        models = ['MOEX_Black_Scholes', 'Classic_Black_Scholes', 'Bachelier_Model', 
                 'Binomial_Tree_With_R', 'Binomial_Tree_Without_R', 
                 'Monte_Carlo_With_R', 'Monte_Carlo_Without_R']
        
        for model in models:
            price = results.get(model, 'N/A')
            if isinstance(price, (int, float)):
                print(f"{model:<25} {price:<10.4f}")
            else:
                print(f"{model:<25} {price:<10}")
        
        # Вывод греков
        print(f"\n\n{'ГРЕКИ':<8} {'MOEX':<12} {'Classic':<12} {'Bachelier':<12}")
        print("-" * 44)
        greeks = ['delta', 'gamma', 'vega', 'theta', 'rho']
        for greek in greeks:
            moex_val = results['Greeks_MOEX'].get(greek, 'N/A') if results['Greeks_MOEX'] else 'N/A'
            classic_val = results['Greeks_Classic'].get(greek, 'N/A') if results['Greeks_Classic'] else 'N/A'
            bachelier_val = results['Greeks_Bachelier'].get(greek, 'N/A') if results['Greeks_Bachelier'] else 'N/A'
            
            if all(isinstance(v, (int, float)) for v in [moex_val, classic_val, bachelier_val] if v != 'N/A'):
                print(f"{greek:<8} {moex_val:<12.6f} {classic_val:<12.6f} {bachelier_val:<12.6f}")
            else:
                print(f"{greek:<8} {moex_val:<12} {classic_val:<12} {bachelier_val:<12}")
        
        # Дополнительная информация
        print(f"\nДополнительная информация:")
        print(f"Время до экспирации: {T_days} дней ({calculator.T:.4f} лет)")
        print(f"Коэффициент S/K: {S/K:.4f}")
        print(f"Волатильность: {iv*100:.2f}%")
        print(f"Безрисковая ставка: {r*100:.2f}%")
        
    except ValueError as e:
        print(f"Ошибка ввода данных: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
    