import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { AboutComponent } from './about/about.component';
import { StockComponent } from './stock/stock.component';
import { ContactComponent } from './contact/contact.component';
import { PlanFreeComponent } from './plan-free/plan-free.component';
import { PlanBasicComponent } from './plan-basic/plan-basic.component';
import { PlanPremiumComponent } from './plan-premium/plan-premium.component';
import { LoginComponent } from './login/login.component';
import { AdminPageComponent } from './admin-page/admin-page.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'about', component: AboutComponent },
  { path: 'stock', component: StockComponent },
  { path: 'contact', component: ContactComponent },
  {path: 'plan-free', component: PlanFreeComponent},
  { path: 'plan-basic', component: PlanBasicComponent },
  { path: 'plan-premium', component: PlanPremiumComponent},
  { path: 'login', component: LoginComponent},
  { path: 'admin-page', component: AdminPageComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
