import { Component } from '@angular/core';

@Component({
  selector: 'app-plan-basic',
  templateUrl: './plan-basic.component.html',
  styleUrls: ['./plan-basic.component.less']
})
export class PlanBasicComponent {
  openColab() {
    window.open('https://colab.research.google.com/drive/1sdB18BDazs2KAT5wUU9wwG8v7NDHbhuE', '_blank');
  }
}
