import { Component } from '@angular/core';
import { Router } from '@angular/router';
@Component({
  selector: 'app-stock',
  templateUrl: './stock.component.html',
  styleUrls: ['./stock.component.less']
})
export class StockComponent {
  selectedPlan: string = "";

  constructor(private router: Router) {}

  onPlanSelected(plan: string): void {
    this.selectedPlan = plan;

    // Redirecionar para o componente adequado com base no plano selecionado
    switch (plan) {
      case 'Gratuito':
        this.router.navigate(['/plan-free']);
        break;
      case 'Básico':
        this.router.navigate(['/plan-basic']);
        break;
      case 'Premium':
        this.router.navigate(['/plan-premium']);
        break;
      default:
        break;
    }
  }
}
