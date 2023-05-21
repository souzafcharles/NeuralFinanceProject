import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PlanPremiumComponent } from './plan-premium.component';

describe('PlanPremiumComponent', () => {
  let component: PlanPremiumComponent;
  let fixture: ComponentFixture<PlanPremiumComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PlanPremiumComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PlanPremiumComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
