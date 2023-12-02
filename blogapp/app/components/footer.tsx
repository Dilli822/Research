import Image from 'next/image';
import Link from 'next/link';
import '@fortawesome/fontawesome-free/css/all.css';
import 'bootstrap/dist/css/bootstrap.min.css';
export default function Footer() {
const currentYear = new Date().getFullYear();
return (
<footer className='mt-3'>
   <div className='container-fluid'>
      <div className='row d-flex align-items-center p-2'>
         <div className='col-6'>
            <div className='d-flex align-items-center'>
               <figure>
                  <Image src='https://unsplash.it/100' alt='logo' className='w-50 h-50' width={50} height={50} />
               </figure>
               <span>
               ©  {currentYear}  Company, Inc
               </span>
            </div>
         </div>
         <div className='col-6'>
            <div className='d-flex justify-content-end'>
               <nav >
                  <ul className='p-auto m-auto'>
                     <li>
                        <Link  href='#'>
                        <i className='fab fa-facebook'></i></Link >
                        {/* Social Media Icons */}
                     </li>
                     <li>
                        <Link href='#'>
                        <i className='fab fa-instagram'> </i></Link> 
                     </li>
                     <li>
                        <Link href='#'>
                        <i className='fab fa-youtube'>  </i></Link>
                     </li>
                  </ul>
               </nav>
            </div>
         </div>
      </div>
   </div>
</footer>
)
}