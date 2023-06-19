import { NgModule } from '@angular/core';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { BrowserModule } from '@angular/platform-browser';
import { FlexLayoutModule } from '@angular/flex-layout';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './header/header.component';
import { HomeComponent } from './home/home.component';
import { FooterComponent } from './footer/footer.component';
import { AboutComponent } from './about/about.component';
import { ContactComponent } from './contact/contact.component';
import { StockComponent } from './stock/stock.component';
import { PlanFreeComponent } from './plan-free/plan-free.component';
import { PlanBasicComponent } from './plan-basic/plan-basic.component';
import { PlanPremiumComponent } from './plan-premium/plan-premium.component';
import { LoginComponent } from './login/login.component';
import { RouterModule } from '@angular/router';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { AdminPageComponent } from './admin-page/admin-page.component';

@NgModule({
  declarations: [AppComponent, HeaderComponent, HomeComponent, FooterComponent, AboutComponent, ContactComponent, StockComponent, PlanFreeComponent, PlanBasicComponent, PlanPremiumComponent, LoginComponent, AdminPageComponent],
  imports: [BrowserModule, AppRoutingModule, NgbModule, FlexLayoutModule, FormsModule, ReactiveFormsModule, RouterModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
