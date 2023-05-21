import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PlanFreeComponent } from './plan-free.component';

describe('PlanFreeComponent', () => {
  let component: PlanFreeComponent;
  let fixture: ComponentFixture<PlanFreeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PlanFreeComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PlanFreeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
