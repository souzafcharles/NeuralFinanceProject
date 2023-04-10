import { Component } from '@angular/core';

@Component({
  selector: 'app-stock',
  templateUrl: './stock.component.html',
  styleUrls: ['./stock.component.less']
})
export class StockComponent {
  selectedPlan = '';
  
  onPlanSelected(plan: string) {
    this.selectedPlan = plan;
  }
  
  // restante do código
}
