import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
import time
from typing import Dict, List, Tuple, Any
import warnings
from dataclasses import dataclass
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuração de estilo para os gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

@dataclass
class SimulationConfig:
    """Configuração para uma simulação específica."""
    num_quartos: int
    num_camareiras: int
    num_supervisoras: int
    taxa_ocupacao: float
    seed: int = 42
    tempo_simulacao: int = 8 * 60  # 8 horas em minutos

@dataclass
class SimulationResult:
    """Resultado de uma simulação."""
    config: SimulationConfig
    avg_turnaround: float
    max_turnaround: float
    min_turnaround: float
    std_turnaround: float
    quartos_processados: int
    quartos_prontos_apos_checkin: int
    utilização_camareiras: float
    utilização_supervisoras: float
    tempo_execucao: float
    turnaround_times: List[float]

class HotelPerformanceAnalyzer:
    """
    Classe principal para análise de performance do hotel.
    Executa múltiplas simulações e gera relatórios detalhados.
    """
    
    def __init__(self, output_dir: str = "resultados_performance"):
        """
        Inicializa o analisador de performance.
        
        Args:
            output_dir: Diretório para salvar os resultados
        """
        self.output_dir = output_dir
        self.results: List[SimulationResult] = []
        self.create_output_directory()
        
        # Parâmetros padrão da simulação original
        self.PROPORCAO_QUARTOS = {'Standard': 0.70, 'Deluxe': 0.25, 'Suite': 0.05}
        self.TEMPO_LIMPEZA = {'Standard': (25, 30), 'Deluxe': (35, 40), 'Suite': (45, 55)}
        self.TEMPO_INSPECAO = (3, 5)
        self.HORA_INICIO_TURNO = 8 * 60
        self.HORA_PICO_CHECKOUT_MEDIA = 11.5 * 60
        self.DESVIO_PADRAO_CHECKOUT = 30
        self.HORA_INICIO_CHECKIN = 14 * 60
    
    def create_output_directory(self):
        """Cria diretório de saída se não existir."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_test_scenarios(self) -> List[SimulationConfig]:
        """
        Cria cenários de teste com diferentes combinações de quartos e funcionários.
        
        Returns:
            Lista de configurações de simulação
        """
        # Definir ranges para testes
        quartos_range = [50, 75, 100, 150, 200]
        camareiras_range = [4, 6, 8, 10, 12]
        supervisoras_range = [1, 2]
        ocupacao_range = [0.65, 0.75, 0.85, 0.95]
        
        scenarios = []
        
        # Teste 1: Variação de quartos com funcionários fixos
        print("Criando cenários - Teste 1: Variação de quartos")
        for quartos in quartos_range:
            scenarios.append(SimulationConfig(
                num_quartos=quartos,
                num_camareiras=8,
                num_supervisoras=1,
                taxa_ocupacao=0.85
            ))
        
        # Teste 2: Variação de camareiras com quartos fixos
        print("Criando cenários - Teste 2: Variação de camareiras")
        for camareiras in camareiras_range:
            scenarios.append(SimulationConfig(
                num_quartos=100,
                num_camareiras=camareiras,
                num_supervisoras=1,
                taxa_ocupacao=0.85
            ))
        
        # Teste 3: Variação de supervisoras
        print("Criando cenários - Teste 3: Variação de supervisoras")
        for supervisoras in supervisoras_range:
            scenarios.append(SimulationConfig(
                num_quartos=100,
                num_camareiras=8,
                num_supervisoras=supervisoras,
                taxa_ocupacao=0.85
            ))
        
        # Teste 4: Variação de ocupação
        print("Criando cenários - Teste 4: Variação de ocupação")
        for ocupacao in ocupacao_range:
            scenarios.append(SimulationConfig(
                num_quartos=100,
                num_camareiras=8,
                num_supervisoras=1,
                taxa_ocupacao=ocupacao
            ))
        
        # Teste 5: Combinações críticas (cenários extremos)
        print("Criando cenários - Teste 5: Cenários extremos")
        critical_scenarios = [
            (200, 4, 1, 0.95),   # Alta demanda, poucos funcionários
            (50, 12, 2, 0.65),   # Baixa demanda, muitos funcionários  
            (150, 6, 1, 0.90),   # Cenário desafiador
            (100, 10, 2, 0.80),  # Cenário bem servido
            (175, 8, 2, 0.85),   # Cenário balanceado expandido
        ]
        
        for quartos, camareiras, supervisoras, ocupacao in critical_scenarios:
            scenarios.append(SimulationConfig(
                num_quartos=quartos,
                num_camareiras=camareiras,
                num_supervisoras=supervisoras,
                taxa_ocupacao=ocupacao
            ))
        
        print(f"Total de cenários criados: {len(scenarios)}")
        return scenarios
    
    def run_single_simulation(self, config: SimulationConfig) -> SimulationResult:
        """
        Executa uma única simulação com a configuração especificada.
        
        Args:
            config: Configuração da simulação
            
        Returns:
            Resultado da simulação
        """
        start_time = time.time()
        
        # Configurar ambiente de simulação
        random.seed(config.seed)
        env = simpy.Environment()
        
        # Classe Hotel adaptada para coletar mais métricas
        class HotelExtended:
            def __init__(self, env, num_camareiras, num_supervisoras):
                self.env = env
                self.camareiras = simpy.Resource(env, capacity=num_camareiras)
                self.supervisora = simpy.Resource(env, capacity=num_supervisoras)
                self.turnaround_times = []
                self.quartos_prontos_apos_checkin = 0
                self.camareira_usage_time = 0
                self.supervisora_usage_time = 0
                self.total_quartos = config.num_quartos
        
            def limpar_e_inspecionar(self, nome_quarto, tipo_quarto, hora_checkout):
                # Processo de limpeza
                with self.camareiras.request() as req_camareira:
                    yield req_camareira
                    tempo_limpeza = random.uniform(*self.TEMPO_LIMPEZA[tipo_quarto])
                    yield self.env.timeout(tempo_limpeza)
                    self.camareira_usage_time += tempo_limpeza
                
                # Processo de inspeção
                with self.supervisora.request() as req_supervisora:
                    yield req_supervisora
                    tempo_inspecao = random.uniform(*self.TEMPO_INSPECAO)
                    yield self.env.timeout(tempo_inspecao)
                    self.supervisora_usage_time += tempo_inspecao
                    
                hora_pronto = self.env.now
                turnaround_time = hora_pronto - hora_checkout
                self.turnaround_times.append(turnaround_time)
                
                if (hora_pronto + self.HORA_INICIO_TURNO) > self.HORA_INICIO_CHECKIN:
                    self.quartos_prontos_apos_checkin += 1
        
        # Adaptar classe Hotel para usar configurações atualizadas
        HotelExtended.TEMPO_LIMPEZA = self.TEMPO_LIMPEZA
        HotelExtended.TEMPO_INSPECAO = self.TEMPO_INSPECAO
        HotelExtended.HORA_INICIO_TURNO = self.HORA_INICIO_TURNO
        HotelExtended.HORA_INICIO_CHECKIN = self.HORA_INICIO_CHECKIN
        
        hotel = HotelExtended(env, config.num_camareiras, config.num_supervisoras)
        
        # Gerador de checkouts adaptado
        def gerador_checkouts(env, hotel, config):
            num_checkouts = int(config.num_quartos * config.taxa_ocupacao)
            
            tipos_a_gerar = []
            for tipo, prop in self.PROPORCAO_QUARTOS.items():
                count = int(round(num_checkouts * prop))
                tipos_a_gerar.extend([tipo] * count)
            
            while len(tipos_a_gerar) < num_checkouts:
                tipos_a_gerar.append('Standard')
            random.shuffle(tipos_a_gerar)

            for i in range(num_checkouts):
                hora_checkout_absoluta = random.normalvariate(
                    self.HORA_PICO_CHECKOUT_MEDIA, 
                    self.DESVIO_PADRAO_CHECKOUT
                )
                hora_checkout_relativa = max(0, hora_checkout_absoluta - self.HORA_INICIO_TURNO)
                
                yield env.timeout(hora_checkout_relativa - env.now if hora_checkout_relativa > env.now else 0)
                
                tipo_quarto = tipos_a_gerar[i]
                nome_quarto = f'Quarto-{i+1:03d}'
                env.process(hotel.limpar_e_inspecionar(nome_quarto, tipo_quarto, env.now))
        
        # Executar simulação
        env.process(gerador_checkouts(env, hotel, config))
        env.run(until=config.tempo_simulacao)
        
        # Calcular métricas
        execution_time = time.time() - start_time
        
        if hotel.turnaround_times:
            avg_turnaround = np.mean(hotel.turnaround_times)
            max_turnaround = np.max(hotel.turnaround_times)
            min_turnaround = np.min(hotel.turnaround_times)
            std_turnaround = np.std(hotel.turnaround_times)
        else:
            avg_turnaround = max_turnaround = min_turnaround = std_turnaround = 0
        
        # Calcular utilização de recursos
        total_time = config.tempo_simulacao
        utilizacao_camareiras = (hotel.camareira_usage_time / (total_time * config.num_camareiras)) * 100
        utilizacao_supervisoras = (hotel.supervisora_usage_time / (total_time * config.num_supervisoras)) * 100
        
        return SimulationResult(
            config=config,
            avg_turnaround=avg_turnaround,
            max_turnaround=max_turnaround,
            min_turnaround=min_turnaround,
            std_turnaround=std_turnaround,
            quartos_processados=len(hotel.turnaround_times),
            quartos_prontos_apos_checkin=hotel.quartos_prontos_apos_checkin,
            utilização_camareiras=min(100, utilizacao_camareiras),
            utilização_supervisoras=min(100, utilizacao_supervisoras),
            tempo_execucao=execution_time,
            turnaround_times=hotel.turnaround_times.copy()
        )
    
    def run_performance_tests(self):
        """Executa os testes de performance usando multiprocessamento para máxima eficiência."""
        scenarios = self.create_test_scenarios()
        total_scenarios = len(scenarios)
        results = []
        print(f"Executando {total_scenarios} cenários de simulação (paralelizados)...")

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.run_single_simulation, config): config for config in scenarios}
            for i, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{i:2d}/{total_scenarios}] OK: {config.num_quartos}Q, {config.num_camareiras}C, {config.num_supervisoras}S, {config.taxa_ocupacao:.0%}")
                except Exception as e:
                    print(f"[{i:2d}/{total_scenarios}] ERRO em {config}: {str(e)}")
        self.results = results

    def plot_turnaround_boxplot(self):
        """Gera um boxplot das distribuições de turnaround (um por cenário)."""
        plt.figure(figsize=(12, 7))
        data = [result.turnaround_times for result in self.results]
        labels = [f"{r.config.num_quartos}Q-{r.config.num_camareiras}C-{r.config.taxa_ocupacao:.0%}" for r in self.results]
        sns.boxplot(data=data)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.title("Distribuição dos Tempos de Turnaround por Cenário")
        plt.xlabel("Cenário")
        plt.ylabel("Turnaround (min)")
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_turnaround_boxplot.png')
        plt.close()

    def plot_pairplot(self, df: pd.DataFrame):
        """Gera um pairplot das principais métricas do sistema."""
        subset = df[["avg_turnaround", "util_camareiras", "util_supervisoras", "quartos", "camareiras", "ocupacao"]]
        sns.pairplot(subset)
        plt.savefig(f'{self.output_dir}/10_pairplot_metrics.png')
        plt.close()

    def validate_result(self, result: SimulationResult) -> bool:
        """
        Valida se os resultados da simulação estão consistentes.        
        """
        # Verificações básicas
        if result.quartos_processados <= 0:
            return False
            
        if result.avg_turnaround <= 0:
            return False
            
        if result.utilização_camareiras > 100 or result.utilização_supervisoras > 100:
            return False
            
        # Verificar se o número de quartos processados é razoável
        expected_quartos = int(result.config.num_quartos * result.config.taxa_ocupacao)
        if abs(result.quartos_processados - expected_quartos) > expected_quartos * 0.1:
            return False
            
        return True
    
    def generate_comprehensive_report(self):
        """Gera relatório abrangente com múltiplos gráficos e análises."""
        if not self.results:
            print("Nenhum resultado disponível para gerar relatórios!")
            return
        
        print("\n=== GERANDO RELATÓRIOS E GRÁFICOS ===")
        
        # Converter resultados para DataFrame
        df = self.create_results_dataframe()
        
        # Gerar diferentes tipos de gráficos
        self.plot_quartos_vs_performance(df)
        self.plot_camareiras_vs_performance(df)
        self.plot_ocupacao_vs_performance(df)
        self.plot_resource_utilization(df)
        self.plot_performance_heatmap(df)
        self.plot_turnaround_distribution()
        self.plot_efficiency_analysis(df)
        
        # Gerar relatório textual
        self.generate_text_report(df)
        
        print(f"📊 Relatórios salvos em: {self.output_dir}")
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Converte resultados em DataFrame para análise."""
        data = []
        for result in self.results:
            data.append({
                'quartos': result.config.num_quartos,
                'camareiras': result.config.num_camareiras,
                'supervisoras': result.config.num_supervisoras,
                'ocupacao': result.config.taxa_ocupacao,
                'avg_turnaround': result.avg_turnaround,
                'max_turnaround': result.max_turnaround,
                'std_turnaround': result.std_turnaround,
                'quartos_processados': result.quartos_processados,
                'quartos_apos_checkin': result.quartos_prontos_apos_checkin,
                'util_camareiras': result.utilização_camareiras,
                'util_supervisoras': result.utilização_supervisoras,
                'tempo_execucao': result.tempo_execucao,
                'eficiencia': result.quartos_processados / result.tempo_execucao if result.tempo_execucao > 0 else 0,
                'ratio_camareira_quarto': result.config.num_camareiras / result.config.num_quartos,
                'quartos_por_camareira': result.config.num_quartos / result.config.num_camareiras
            })
        
        return pd.DataFrame(data)
    
    def plot_quartos_vs_performance(self, df: pd.DataFrame):
        """Gráfico: Impacto do número de quartos no desempenho."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filtrar dados com parâmetros fixos
        df_fixed = df[(df['camareiras'] == 8) & (df['supervisoras'] == 1) & (df['ocupacao'] == 0.85)]
        
        if not df_fixed.empty:
            # Turnaround médio vs quartos
            ax1.plot(df_fixed['quartos'], df_fixed['avg_turnaround'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Número de Quartos')
            ax1.set_ylabel('Turnaround Médio (min)')
            ax1.set_title('Turnaround Médio vs Número de Quartos')
            ax1.grid(True, alpha=0.3)
            
            # Utilização de camareiras vs quartos
            ax2.plot(df_fixed['quartos'], df_fixed['util_camareiras'], 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Número de Quartos')
            ax2.set_ylabel('Utilização Camareiras (%)')
            ax2.set_title('Utilização de Camareiras vs Número de Quartos')
            ax2.grid(True, alpha=0.3)
            
            # Quartos processados vs quartos totais
            ax3.plot(df_fixed['quartos'], df_fixed['quartos_processados'], 'go-', linewidth=2, markersize=8)
            ax3.set_xlabel('Número de Quartos')
            ax3.set_ylabel('Quartos Processados')
            ax3.set_title('Quartos Processados vs Quartos Totais')
            ax3.grid(True, alpha=0.3)
            
            # Quartos prontos após check-in
            ax4.plot(df_fixed['quartos'], df_fixed['quartos_apos_checkin'], 'mo-', linewidth=2, markersize=8)
            ax4.set_xlabel('Número de Quartos')
            ax4.set_ylabel('Quartos Prontos Após 14:00')
            ax4.set_title('Quartos Não Prontos para Check-in')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_quartos_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_camareiras_vs_performance(self, df: pd.DataFrame):
        """Gráfico: Impacto do número de camareiras no desempenho."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filtrar dados com parâmetros fixos
        df_fixed = df[(df['quartos'] == 100) & (df['supervisoras'] == 1) & (df['ocupacao'] == 0.85)]
        
        if not df_fixed.empty:
            # Turnaround médio vs camareiras
            ax1.plot(df_fixed['camareiras'], df_fixed['avg_turnaround'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Número de Camareiras')
            ax1.set_ylabel('Turnaround Médio (min)')
            ax1.set_title('Turnaround Médio vs Número de Camareiras')
            ax1.grid(True, alpha=0.3)
            
            # Utilização de camareiras
            ax2.plot(df_fixed['camareiras'], df_fixed['util_camareiras'], 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('Número de Camareiras')
            ax2.set_ylabel('Utilização Camareiras (%)')
            ax2.set_title('Utilização de Camareiras')
            ax2.grid(True, alpha=0.3)
            
            # Eficiência (quartos processados por unidade de tempo)
            ax3.plot(df_fixed['camareiras'], df_fixed['eficiencia'], 'go-', linewidth=2, markersize=8)
            ax3.set_xlabel('Número de Camareiras')
            ax3.set_ylabel('Eficiência (quartos/min)')
            ax3.set_title('Eficiência do Sistema')
            ax3.grid(True, alpha=0.3)
            
            # Quartos por camareira
            ax4.bar(df_fixed['camareiras'], df_fixed['quartos_por_camareira'], alpha=0.7, color='orange')
            ax4.set_xlabel('Número de Camareiras')
            ax4.set_ylabel('Quartos por Camareira')
            ax4.set_title('Carga de Trabalho por Camareira')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_camareiras_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ocupacao_vs_performance(self, df: pd.DataFrame):
        """Gráfico: Impacto da taxa de ocupação no desempenho."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filtrar dados com parâmetros fixos
        df_fixed = df[(df['quartos'] == 100) & (df['camareiras'] == 8) & (df['supervisoras'] == 1)]
        
        if not df_fixed.empty:
            # Turnaround médio vs ocupação
            ax1.plot(df_fixed['ocupacao'] * 100, df_fixed['avg_turnaround'], 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Taxa de Ocupação (%)')
            ax1.set_ylabel('Turnaround Médio (min)')
            ax1.set_title('Turnaround Médio vs Taxa de Ocupação')
            ax1.grid(True, alpha=0.3)
            
            # Utilização vs ocupação
            ax2.plot(df_fixed['ocupacao'] * 100, df_fixed['util_camareiras'], 'ro-', 
                    linewidth=2, markersize=8, label='Camareiras')
            ax2.plot(df_fixed['ocupacao'] * 100, df_fixed['util_supervisoras'], 'go-', 
                    linewidth=2, markersize=8, label='Supervisoras')
            ax2.set_xlabel('Taxa de Ocupação (%)')
            ax2.set_ylabel('Utilização (%)')
            ax2.set_title('Utilização de Recursos vs Ocupação')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Quartos processados vs ocupação
            ax3.plot(df_fixed['ocupacao'] * 100, df_fixed['quartos_processados'], 'mo-', linewidth=2, markersize=8)
            ax3.set_xlabel('Taxa de Ocupação (%)')
            ax3.set_ylabel('Quartos Processados')
            ax3.set_title('Quartos Processados vs Ocupação')
            ax3.grid(True, alpha=0.3)
            
            # Desvio padrão do turnaround
            ax4.plot(df_fixed['ocupacao'] * 100, df_fixed['std_turnaround'], 'co-', linewidth=2, markersize=8)
            ax4.set_xlabel('Taxa de Ocupação (%)')
            ax4.set_ylabel('Desvio Padrão Turnaround (min)')
            ax4.set_title('Variabilidade do Turnaround vs Ocupação')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_ocupacao_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_resource_utilization(self, df: pd.DataFrame):
        """Gráfico: Análise de utilização de recursos."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Utilização de camareiras vs supervisoras
        scatter = ax1.scatter(df['util_camareiras'], df['util_supervisoras'], 
                             c=df['avg_turnaround'], s=df['quartos']*2, 
                             alpha=0.6, cmap='viridis')
        ax1.set_xlabel('Utilização Camareiras (%)')
        ax1.set_ylabel('Utilização Supervisoras (%)')
        ax1.set_title('Utilização de Recursos\n(Tamanho = Nº Quartos, Cor = Turnaround)')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar linha de referência de 80% de utilização
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Meta 80%')
        ax1.axvline(x=80, color='red', linestyle='--', alpha=0.7)
        ax1.legend()
        
        plt.colorbar(scatter, ax=ax1, label='Turnaround Médio (min)')
        
        # Análise de eficiência por configuração
        df_efficiency = df.groupby(['camareiras', 'supervisoras']).agg({
            'avg_turnaround': 'mean',
            'util_camareiras': 'mean',
            'eficiencia': 'mean'
        }).reset_index()
        
        df_efficiency['config'] = df_efficiency['camareiras'].astype(str) + 'C/' + df_efficiency['supervisoras'].astype(str) + 'S'
        
        bars = ax2.bar(df_efficiency['config'], df_efficiency['eficiencia'], alpha=0.7)
        ax2.set_xlabel('Configuração (Camareiras/Supervisoras)')
        ax2.set_ylabel('Eficiência (quartos/min)')
        ax2.set_title('Eficiência por Configuração de Pessoal')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, efficiency in zip(bars, df_efficiency['eficiencia']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{efficiency:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_resource_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self, df: pd.DataFrame):
        """Gráfico: Mapa de calor do desempenho."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Preparar dados para heatmap
        # Turnaround médio por quartos e camareiras
        pivot_turnaround = df.pivot_table(values='avg_turnaround', index='camareiras', 
                                         columns='quartos', aggfunc='mean')
        
        if not pivot_turnaround.empty:
            sns.heatmap(pivot_turnaround, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax1)
            ax1.set_title('Turnaround Médio (min)\nQuartos vs Camareiras')
            ax1.set_ylabel('Camareiras')
            ax1.set_xlabel('Quartos')
        
        # Utilização de camareiras
        pivot_util = df.pivot_table(values='util_camareiras', index='camareiras', 
                                   columns='quartos', aggfunc='mean')
        
        if not pivot_util.empty:
            sns.heatmap(pivot_util, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2)
            ax2.set_title('Utilização Camareiras (%)\nQuartos vs Camareiras')
            ax2.set_ylabel('Camareiras')
            ax2.set_xlabel('Quartos')
        
        # Quartos prontos após check-in (problema)
        pivot_late = df.pivot_table(values='quartos_apos_checkin', index='camareiras', 
                                   columns='quartos', aggfunc='mean')
        
        if not pivot_late.empty:
            sns.heatmap(pivot_late, annot=True, fmt='.0f', cmap='Reds', ax=ax3)
            ax3.set_title('Quartos Não Prontos p/ Check-in\nQuartos vs Camareiras')
            ax3.set_ylabel('Camareiras')
            ax3.set_xlabel('Quartos')
        
        # Eficiência geral
        pivot_efficiency = df.pivot_table(values='eficiencia', index='camareiras', 
                                         columns='quartos', aggfunc='mean')
        
        if not pivot_efficiency.empty:
            sns.heatmap(pivot_efficiency, annot=True, fmt='.3f', cmap='Blues', ax=ax4)
            ax4.set_title('Eficiência (quartos/min)\nQuartos vs Camareiras')
            ax4.set_ylabel('Camareiras')
            ax4.set_xlabel('Quartos')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_turnaround_distribution(self):
        """Gráfico: Distribuição dos tempos de turnaround por cenário."""
        # Selecionar alguns cenários representativos
        selected_results = []
        
        # Cenário padrão
        for result in self.results:
            config = result.config
            if (config.num_quartos == 100 and config.num_camareiras == 8 and 
                config.num_supervisoras == 1 and config.taxa_ocupacao == 0.85):
                selected_results.append(("Cenário Padrão\n(100Q, 8C, 1S, 85%)", result))
                break
        
        # Cenário com poucos funcionários
        for result in self.results:
            config = result.config
            if (config.num_quartos == 200 and config.num_camareiras == 4 and 
                config.num_supervisoras == 1 and config.taxa_ocupacao == 0.95):
                selected_results.append(("Alta Demanda\n(200Q, 4C, 1S, 95%)", result))
                break
        
        # Cenário bem servido
        for result in self.results:
            config = result.config
            if (config.num_quartos == 100 and config.num_camareiras == 10 and 
                config.num_supervisoras == 2 and config.taxa_ocupacao == 0.80):
                selected_results.append(("Bem Servido\n(100Q, 10C, 2S, 80%)", result))
                break
        
        if selected_results:
            fig, axes = plt.subplots(1, len(selected_results), figsize=(5*len(selected_results), 6))
            if len(selected_results) == 1:
                axes = [axes]
            
            for i, (label, result) in enumerate(selected_results):
                if result.turnaround_times:
                    axes[i].hist(result.turnaround_times, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].axvline(np.mean(result.turnaround_times), color='red', 
                                   linestyle='--', linewidth=2, label=f'Média: {np.mean(result.turnaround_times):.1f}min')
                    axes[i].axvline(75, color='green', linestyle='--', linewidth=2, 
                                   label='Meta: 75min')
                    axes[i].set_title(label)
                    axes[i].set_xlabel('Tempo de Turnaround (min)')
                    axes[i].set_ylabel('Frequência')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/06_turnaround_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_efficiency_analysis(self, df: pd.DataFrame):
        """Gráfico: Análise de eficiência e custos."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Assumindo custos hipotéticos
        df['custo_camareiras'] = df['camareiras'] * 20  # $20/hora por camareira
        df['custo_supervisoras'] = df['supervisoras'] * 30  # $30/hora por supervisora
        df['custo_total'] = df['custo_camareiras'] + df['custo_supervisoras']
        df['custo_por_quarto'] = df['custo_total'] / df['quartos_processados']
        
        # Eficiência vs Custo
        scatter = ax1.scatter(df['custo_total'], df['eficiencia'], 
                             c=df['avg_turnaround'], s=100, alpha=0.6, cmap='viridis')
        ax1.set_xlabel('Custo Total por Turno ($)')
        ax1.set_ylabel('Eficiência (quartos/min)')
        ax1.set_title('Eficiência vs Custo Total')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Turnaround Médio (min)')
        
        # Custo por quarto vs Turnaround
        ax2.scatter(df['custo_por_quarto'], df['avg_turnaround'], alpha=0.6, s=100)
        ax2.set_xlabel('Custo por Quarto Processado ($)')
        ax2.set_ylabel('Turnaround Médio (min)')
        ax2.set_title('Custo vs Qualidade do Serviço')
        ax2.grid(True, alpha=0.3)
        
        # Análise de ponto ótimo
        df['score_efficiency'] = (1 / df['avg_turnaround']) * df['eficiencia'] / df['custo_por_quarto']
        
        top_configs = df.nlargest(10, 'score_efficiency')
        
        bars = ax3.bar(range(len(top_configs)), top_configs['score_efficiency'], alpha=0.7)
        ax3.set_xlabel('Configurações (Top 10)')
        ax3.set_ylabel('Score de Eficiência')
        ax3.set_title('Top 10 Configurações Mais Eficientes')
        ax3.grid(True, alpha=0.3)
        
        # Análise de ROI (Return on Investment)
        df['roi'] = df['quartos_processados'] / df['custo_total'] * 100  # Quartos por dólar
        
        ax4.scatter(df['roi'], df['avg_turnaround'], c=df['util_camareiras'], 
                   s=100, alpha=0.6, cmap='coolwarm')
        ax4.set_xlabel('ROI (quartos/$)')
        ax4.set_ylabel('Turnaround Médio (min)')
        ax4.set_title('ROI vs Qualidade do Serviço')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_text_report(self, df: pd.DataFrame):
        """Gera relatório textual detalhado."""
        report_path = f'{self.output_dir}/relatorio_performance.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE PERFORMANCE - SIMULAÇÃO HOTELEIRA\n")
            f.write("="*80 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de simulações executadas: {len(self.results)}\n\n")
            
            # Estatísticas gerais
            f.write("ESTATÍSTICAS GERAIS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Turnaround médio geral: {df['avg_turnaround'].mean():.2f} ± {df['avg_turnaround'].std():.2f} min\n")
            f.write(f"Turnaround mínimo observado: {df['avg_turnaround'].min():.2f} min\n")
            f.write(f"Turnaround máximo observado: {df['avg_turnaround'].max():.2f} min\n")
            f.write(f"Utilização média de camareiras: {df['util_camareiras'].mean():.1f}%\n")
            f.write(f"Utilização média de supervisoras: {df['util_supervisoras'].mean():.1f}%\n\n")
            
            # Melhores configurações
            f.write("TOP 5 MELHORES CONFIGURAÇÕES (menor turnaround)\n")
            f.write("-" * 50 + "\n")
            best_configs = df.nsmallest(5, 'avg_turnaround')
            for i, (_, row) in enumerate(best_configs.iterrows(), 1):
                f.write(f"{i}. {row['quartos']} quartos, {row['camareiras']} camareiras, "
                       f"{row['supervisoras']} supervisoras, {row['ocupacao']:.0%} ocupação\n")
                f.write(f"   Turnaround: {row['avg_turnaround']:.2f} min, "
                       f"Utilização: {row['util_camareiras']:.1f}%\n")
            
            f.write("\nTOP 5 PIORES CONFIGURAÇÕES (maior turnaround)\n")
            f.write("-" * 50 + "\n")
            worst_configs = df.nlargest(5, 'avg_turnaround')
            for i, (_, row) in enumerate(worst_configs.iterrows(), 1):
                f.write(f"{i}. {row['quartos']} quartos, {row['camareiras']} camareiras, "
                       f"{row['supervisoras']} supervisoras, {row['ocupacao']:.0%} ocupação\n")
                f.write(f"   Turnaround: {row['avg_turnaround']:.2f} min, "
                       f"Utilização: {row['util_camareiras']:.1f}%\n")
            
            # Análises específicas
            f.write("\nANÁLISES ESPECÍFICAS\n")
            f.write("-" * 50 + "\n")
            
            # Impacto do número de quartos
            if len(df[df['camareiras'] == 8]) > 1:
                correlation_quartos = df[df['camareiras'] == 8]['quartos'].corr(df[df['camareiras'] == 8]['avg_turnaround'])
                f.write(f"Correlação entre nº de quartos e turnaround: {correlation_quartos:.3f}\n")
            
            # Impacto do número de camareiras
            if len(df[df['quartos'] == 100]) > 1:
                correlation_camareiras = df[df['quartos'] == 100]['camareiras'].corr(df[df['quartos'] == 100]['avg_turnaround'])
                f.write(f"Correlação entre nº de camareiras e turnaround: {correlation_camareiras:.3f}\n")
            
            # Problemas identificados
            f.write("\nPROBLEMAS IDENTIFICADOS\n")
            f.write("-" * 50 + "\n")
            
            high_turnaround = df[df['avg_turnaround'] > 90]
            if not high_turnaround.empty:
                f.write(f"{len(high_turnaround)} configurações com turnaround > 90 min\n")
            
            low_utilization = df[df['util_camareiras'] < 60]
            if not low_utilization.empty:
                f.write(f"{len(low_utilization)} configurações com baixa utilização de camareiras (< 60%)\n")
            
            over_utilization = df[df['util_camareiras'] > 95]
            if not over_utilization.empty:
                f.write(f"{len(over_utilization)} configurações com super utilização de camareiras (> 95%)\n")
            
            late_rooms = df[df['quartos_apos_checkin'] > df['quartos_processados'] * 0.1]
            if not late_rooms.empty:
                f.write(f"{len(late_rooms)} configurações com >10% dos quartos não prontos para check-in\n")
            
            # Recomendações
            f.write("\nRECOMENDAÇÕES\n")
            f.write("-" * 50 + "\n")
            
            optimal_config = df.loc[df['avg_turnaround'].idxmin()]
            f.write(f"Configuração ótima encontrada:\n")
            f.write(f"   - {optimal_config['quartos']} quartos\n")
            f.write(f"   - {optimal_config['camareiras']} camareiras\n")
            f.write(f"   - {optimal_config['supervisoras']} supervisoras\n")
            f.write(f"   - {optimal_config['ocupacao']:.0%} ocupação\n")
            f.write(f"   - Turnaround: {optimal_config['avg_turnaround']:.2f} min\n")
            
            # Razão quartos/camareiras ideal
            efficient_configs = df[df['avg_turnaround'] <= df['avg_turnaround'].quantile(0.25)]
            avg_ratio = efficient_configs['quartos_por_camareira'].mean()
            f.write(f"\nRazão quartos/camareira recomendada: ~{avg_ratio:.1f} quartos por camareira\n")
            
            f.write(f"\nPara hotéis com alta ocupação (>85%), recomenda-se:\n")
            high_occupancy = df[df['ocupacao'] >= 0.85]
            if not high_occupancy.empty:
                best_high_occ = high_occupancy.loc[high_occupancy['avg_turnaround'].idxmin()]
                f.write(f"   - Mínimo {best_high_occ['camareiras']} camareiras por 100 quartos\n")
                f.write(f"   - {best_high_occ['supervisoras']} supervisoras\n")
        
        print(f"Relatório textual salvo em: {report_path}")
    
    def save_raw_data(self):
        """Salva dados brutos em formato CSV e JSON."""
        # Salvar como CSV
        df = self.create_results_dataframe()
        csv_path = f'{self.output_dir}/dados_simulacao.csv'
        df.to_csv(csv_path, index=False)
        
        # Salvar como JSON (com mais detalhes)
        json_data = []
        for result in self.results:
            json_data.append({
                'config': {
                    'quartos': result.config.num_quartos,
                    'camareiras': result.config.num_camareiras,
                    'supervisoras': result.config.num_supervisoras,
                    'ocupacao': result.config.taxa_ocupacao,
                    'seed': result.config.seed,
                    'tempo_simulacao': result.config.tempo_simulacao
                },
                'metrics': {
                    'avg_turnaround': result.avg_turnaround,
                    'max_turnaround': result.max_turnaround,
                    'min_turnaround': result.min_turnaround,
                    'std_turnaround': result.std_turnaround,
                    'quartos_processados': result.quartos_processados,
                    'quartos_prontos_apos_checkin': result.quartos_prontos_apos_checkin,
                    'utilizacao_camareiras': result.utilização_camareiras,
                    'utilizacao_supervisoras': result.utilização_supervisoras,
                    'tempo_execucao': result.tempo_execucao
                },
                'turnaround_times': result.turnaround_times
            })
        
        json_path = f'{self.output_dir}/dados_simulacao.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"💾 Dados salvos em: {csv_path} e {json_path}")

def main():
    """Função principal para executar todos os testes."""
    print("SISTEMA DE ANÁLISE DE PERFORMANCE HOTELEIRA")
    print("=" * 60)
    print("Este sistema executará uma bateria completa de testes")
    print("para avaliar o impacto de diferentes configurações")
    print("no desempenho operacional do hotel.\n")
    
    # Inicializar analisador
    analyzer = HotelPerformanceAnalyzer()
    
    try:
        # Executar testes
        start_time = time.time()
        analyzer.run_performance_tests()
        
        # Gerar relatórios
        analyzer.generate_comprehensive_report()
        
        # Salvar dados brutos
        analyzer.save_raw_data()
        
        total_time = time.time() - start_time
        
        print(f"\nANÁLISE COMPLETA FINALIZADA!")
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        print(f"Resultados disponíveis em: {analyzer.output_dir}")
        print("\nGráficos gerados:")
        print("   01_quartos_vs_performance.png - Impacto do número de quartos")
        print("   02_camareiras_vs_performance.png - Impacto do número de camareiras")
        print("   03_ocupacao_vs_performance.png - Impacto da taxa de ocupação")
        print("   04_resource_utilization.png - Análise de utilização de recursos")
        print("   05_performance_heatmap.png - Mapas de calor de performance")
        print("   06_turnaround_distributions.png - Distribuições de turnaround")
        print("   07_efficiency_analysis.png - Análise de eficiência e custos")
        print("\nRelatórios:")
        print("   relatorio_performance.txt - Relatório detalhado")
        print("   dados_simulacao.csv - Dados em formatoa planilha")
        print("   dados_simulacao.json - Dados estruturados")
        
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()
