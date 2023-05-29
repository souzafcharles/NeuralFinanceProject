import { Component } from '@angular/core';

@Component({
  selector: 'app-plan-premium',
  templateUrl: './plan-premium.component.html',
  styleUrls: ['./plan-premium.component.less']
})
export class PlanPremiumComponent {
  openColab() {
    window.open('https://colab.research.google.com/drive/1sdB18BDazs2KAT5wUU9wwG8v7NDHbhuE', '_blank');
  }
}
