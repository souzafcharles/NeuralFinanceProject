import { Component } from '@angular/core';

@Component({
  selector: 'app-plan-free',
  templateUrl: './plan-free.component.html',
  styleUrls: ['./plan-free.component.less']
})
export class PlanFreeComponent {
  openColab() {
    window.open('https://colab.research.google.com/drive/1sdB18BDazs2KAT5wUU9wwG8v7NDHbhuE', '_blank');
  }
}
