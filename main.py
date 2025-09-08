import simpy
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt

# --- Parâmetros Padrão da Simulação ---
# Estes valores serão usados se nenhum argumento for fornecido via CLI.
RANDOM_SEED = 42
NUM_CAMAREIRAS = 8
NUM_SUPERVISORAS = 1
TOTAL_QUARTOS = 100
TAXA_OCUPACAO_DIA = 0.85
PROPORCAO_QUARTOS = {'Standard': 0.70, 'Deluxe': 0.25, 'Suite': 0.05}
TEMPO_LIMPEZA = {'Standard': (25, 30), 'Deluxe': (35, 40), 'Suite': (45, 55)}
TEMPO_INSPECAO = (3, 5)
HORA_INICIO_TURNO = 8 * 60
HORA_PICO_CHECKOUT_MEDIA = 11.5 * 60
DESVIO_PADRAO_CHECKOUT = 30
HORA_INICIO_CHECKIN = 14 * 60
TEMPO_SIMULACAO = 8 * 60

class Hotel:
    """Classe que representa o ambiente do hotel, contendo os recursos e processos."""
    def __init__(self, env, num_camareiras, num_supervisoras):
        self.env = env
        self.camareiras = simpy.Resource(env, capacity=num_camareiras)
        self.supervisora = simpy.Resource(env, capacity=num_supervisoras)
        self.turnaround_times = []
        self.quartos_prontos_apos_checkin = 0

    def limpar_e_inspecionar(self, nome_quarto, tipo_quarto, hora_checkout):
        """Processo que simula o ciclo de vida de um quarto desde o checkout até estar pronto."""
        with self.camareiras.request() as req_camareira:
            yield req_camareira
            yield self.env.timeout(random.uniform(*TEMPO_LIMPEZA[tipo_quarto]))
        
        with self.supervisora.request() as req_supervisora:
            yield req_supervisora
            yield self.env.timeout(random.uniform(*TEMPO_INSPECAO))
            
        hora_pronto = self.env.now
        turnaround_time = hora_pronto - hora_checkout
        self.turnaround_times.append(turnaround_time)
        
        if (hora_pronto + HORA_INICIO_TURNO) > HORA_INICIO_CHECKIN:
            self.quartos_prontos_apos_checkin += 1

def gerador_checkouts(env, hotel, taxa_ocupacao):
    """Processo que gera os checkouts dos quartos ao longo do dia."""
    num_checkouts = int(TOTAL_QUARTOS * taxa_ocupacao)
    
    tipos_a_gerar = []
    for tipo, prop in PROPORCAO_QUARTOS.items():
        count = int(round(num_checkouts * prop))
        tipos_a_gerar.extend([tipo] * count)
    
    while len(tipos_a_gerar) < num_checkouts:
        tipos_a_gerar.append('Standard')
    random.shuffle(tipos_a_gerar)

    for i in range(num_checkouts):
        hora_checkout_absoluta = random.normalvariate(HORA_PICO_CHECKOUT_MEDIA, DESVIO_PADRAO_CHECKOUT)
        hora_checkout_relativa = max(0, hora_checkout_absoluta - HORA_INICIO_TURNO)
        
        yield env.timeout(hora_checkout_relativa - env.now if hora_checkout_relativa > env.now else 0)
        
        tipo_quarto = tipos_a_gerar[i]
        nome_quarto = f'Quarto-{i+1:03d}'
        env.process(hotel.limpar_e_inspecionar(nome_quarto, tipo_quarto, env.now))

def run_simulation(num_camareiras, taxa_ocupacao, seed, plot):
    """Executa uma rodada da simulação e exibe os resultados."""
    print(f"\n--- Iniciando Simulação com {num_camareiras} camareira(s) e ocupação de {taxa_ocupacao*100:.0f}% ---")
    random.seed(seed)
    env = simpy.Environment()
    hotel = Hotel(env, num_camareiras, NUM_SUPERVISORAS)
    env.process(gerador_checkouts(env, hotel, taxa_ocupacao))
    env.run(until=TEMPO_SIMULACAO)

    # --- Apresentação dos Resultados ---
    print("\n--- Resultados da Simulação ---")
    total_quartos_processados = len(hotel.turnaround_times)
    print(f"Total de quartos processados: {total_quartos_processados}")

    if total_quartos_processados > 0:
        avg_turnaround = np.mean(hotel.turnaround_times)
        max_turnaround = np.max(hotel.turnaround_times)
        
        print(f"  - Tempo Médio de Turnaround: {avg_turnaround:.2f} minutos")
        print(f"  - Tempo Máximo de Turnaround: {max_turnaround:.2f} minutos")
        print(f"  - Quartos prontos APÓS 14:00: {hotel.quartos_prontos_apos_checkin}")

        if plot:
            plot_results(hotel.turnaround_times, num_camareiras, taxa_ocupacao)
    else:
        print("Nenhum quarto foi processado.")

def plot_results(turnaround_times, num_camareiras, taxa_ocupacao):
    """Gera e exibe um histograma dos tempos de turnaround."""
    plt.figure(figsize=(10, 6))
    plt.hist(turnaround_times, bins=20, edgecolor='black', alpha=0.7)
    
    plt.title(f'Distribuição do Tempo de Turnaround\n({num_camareiras} Camareiras, {taxa_ocupacao*100:.0f}% Ocupação)')
    plt.xlabel('Tempo de Turnaround (minutos)')
    plt.ylabel('Número de Quartos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Linha vertical para indicar a meta de turnaround
    plt.axvline(x=75, color='r', linestyle='--', linewidth=2, label='Meta de Turnaround (75 min)')
    
    avg_turnaround = np.mean(turnaround_times)
    plt.axvline(x=avg_turnaround, color='g', linestyle='-', linewidth=2, label=f'Média ({avg_turnaround:.2f} min)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulador de Processo de Limpeza de Hotel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--camareiras", 
        type=int, 
        default=NUM_CAMAREIRAS,
        help="Número de camareiras no turno."
    )
    parser.add_argument(
        "-o", "--ocupacao", 
        type=float, 
        default=TAXA_OCUPACAO_DIA,
        help="Taxa de ocupação do hotel (ex: 0.85 para 85%%)."
    )
    parser.add_argument(
        "-s", "--seed", 
        type=int, 
        default=RANDOM_SEED,
        help="Semente para o gerador de números aleatórios."
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Exibir um gráfico com os resultados ao final da simulação."
    )
    
    args = parser.parse_args()

    run_simulation(
        num_camareiras=args.camareiras,
        taxa_ocupacao=args.ocupacao,
        seed=args.seed,
        plot=args.plot
    )

"""
### Como Utilizar este script
1. Rodar o cenário base e mostrar o gráfico: python main.py --plot
2. Rodar o cenário A (9 camareiras) e mostrar o gráfico: python main.py --camareiras 9 --plot
3. Simular um dia com baixa ocupação (60%): python main.py --ocupacao 0.60 --plot
4. Ver todas as opções disponíveis: python main.py --help
"""